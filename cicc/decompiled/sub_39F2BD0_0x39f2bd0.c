// Function: sub_39F2BD0
// Address: 0x39f2bd0
//
void __fastcall sub_39F2BD0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rdx
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rbx
  _BYTE *v9; // r8
  size_t v10; // r13
  _BYTE *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 *v14; // r12
  _BYTE *v15; // [rsp+8h] [rbp-68h]
  size_t v16; // [rsp+18h] [rbp-58h] BYREF
  __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  __int64 v18; // [rsp+28h] [rbp-48h]
  unsigned __int64 v19; // [rsp+30h] [rbp-40h]

  v3 = 32 * a3;
  v4 = a2 + v3;
  v17 = 0;
  v5 = *(_QWORD *)(a1 + 264);
  v18 = 0;
  v19 = 0;
  if ( v3 > 0x7FFFFFFFFFFFFFE0LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v6 = v3;
  if ( v3 )
  {
    v7 = a2;
    v17 = sub_22077B0(v3);
    v8 = v17;
    v19 = v17 + v6;
    if ( a2 != v4 )
    {
      while ( 1 )
      {
        if ( !v8 )
          goto LABEL_7;
        v9 = *(_BYTE **)v7;
        v10 = *(_QWORD *)(v7 + 8);
        v11 = (_BYTE *)(v8 + 16);
        *(_QWORD *)v8 = v8 + 16;
        if ( &v9[v10] && !v9 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v16 = v10;
        if ( v10 > 0xF )
          break;
        if ( v10 == 1 )
        {
          *(_BYTE *)(v8 + 16) = *v9;
        }
        else if ( v10 )
        {
          goto LABEL_16;
        }
LABEL_6:
        *(_QWORD *)(v8 + 8) = v10;
        v11[v10] = 0;
LABEL_7:
        v7 += 32;
        v8 += 32;
        if ( v4 == v7 )
          goto LABEL_18;
      }
      v15 = v9;
      v12 = sub_22409D0(v8, &v16, 0);
      v9 = v15;
      *(_QWORD *)v8 = v12;
      v11 = (_BYTE *)v12;
      *(_QWORD *)(v8 + 16) = v16;
LABEL_16:
      memcpy(v11, v9, v10);
      v10 = v16;
      v11 = *(_BYTE **)v8;
      goto LABEL_6;
    }
  }
  else
  {
    v8 = 0;
  }
LABEL_18:
  v18 = v8;
  v13 = *(_QWORD *)(v5 + 136);
  if ( v13 == *(_QWORD *)(v5 + 144) )
  {
    sub_E83790((char **)(v5 + 128), (char *)v13, &v17);
    v8 = v18;
    v14 = (unsigned __int64 *)v17;
  }
  else
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = v17;
      *(_QWORD *)(v13 + 8) = v18;
      *(_QWORD *)(v13 + 16) = v19;
      *(_QWORD *)(v5 + 136) += 24LL;
      return;
    }
    *(_QWORD *)(v5 + 136) = 24;
    v14 = (unsigned __int64 *)v17;
  }
  if ( (unsigned __int64 *)v8 != v14 )
  {
    do
    {
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        j_j___libc_free_0(*v14);
      v14 += 4;
    }
    while ( v14 != (unsigned __int64 *)v8 );
    v14 = (unsigned __int64 *)v17;
  }
  if ( v14 )
    j_j___libc_free_0((unsigned __int64)v14);
}
