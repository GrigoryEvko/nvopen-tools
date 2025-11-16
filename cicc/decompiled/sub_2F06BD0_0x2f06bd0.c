// Function: sub_2F06BD0
// Address: 0x2f06bd0
//
_QWORD *__fastcall sub_2F06BD0(_QWORD *a1, const void *a2, __int64 a3, char a4)
{
  _QWORD *v7; // rax
  char v8; // bl
  _QWORD *v9; // r14
  unsigned __int64 v10; // r13
  char *v11; // rcx
  char *v12; // rax
  char *v13; // [rsp+8h] [rbp-38h]

  if ( (_BYTE)qword_5022408 )
  {
    v7 = (_QWORD *)sub_22077B0(0x28u);
    v8 = a4 ^ 1;
    v9 = v7;
    if ( v7 )
    {
      v10 = 8 * a3;
      v7[1] = 0;
      *v7 = off_4A2A480;
      v7[2] = 0;
      v7[3] = 0;
      if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v11 = 0;
      if ( v10 )
      {
        v12 = (char *)sub_22077B0(v10);
        v9[1] = v12;
        v9[3] = &v12[v10];
        v13 = &v12[v10];
        memcpy(v12, a2, v10);
        v11 = v13;
      }
      v9[2] = v11;
      *((_BYTE *)v9 + 32) = v8;
    }
    *a1 = v9;
  }
  else
  {
    *a1 = 0;
  }
  return a1;
}
