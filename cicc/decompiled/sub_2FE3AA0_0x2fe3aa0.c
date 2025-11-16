// Function: sub_2FE3AA0
// Address: 0x2fe3aa0
//
char __fastcall sub_2FE3AA0(__int64 a1, __int64 **a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rax
  _QWORD *v4; // r14
  _QWORD *v5; // rbx
  _DWORD *v6; // rdi
  __int64 v7; // rdx
  __int64 v9; // [rsp+8h] [rbp-58h]
  const char *v10; // [rsp+10h] [rbp-50h] BYREF
  char v11; // [rsp+30h] [rbp-30h]
  char v12; // [rsp+31h] [rbp-2Fh]

  v2 = sub_BA8B30((__int64)a2, (__int64)"__stack_chk_guard", 0x11u);
  if ( !v2 )
  {
    v3 = sub_BCE3C0(*a2, 0);
    v10 = "__stack_chk_guard";
    v4 = (_QWORD *)v3;
    v12 = 1;
    v11 = 3;
    BYTE4(v9) = 0;
    v5 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v5 )
      sub_B30000((__int64)v5, (__int64)a2, v4, 0, 0, 0, (__int64)&v10, 0, 0, v9, 0);
    LOBYTE(v2) = sub_BAA7B0((__int64)a2);
    if ( (_BYTE)v2 )
    {
      v6 = *(_DWORD **)(a1 + 8);
      v2 = (unsigned int)v6[139];
      if ( (_DWORD)v2 == 14 )
      {
        if ( v6[140] == 1 )
          return v2;
      }
      else
      {
        if ( (unsigned int)(v6[136] - 24) <= 1 && (_DWORD)v2 == 3 )
          return v2;
        if ( (unsigned int)v2 > 0x1F )
        {
LABEL_10:
          *((_BYTE *)v5 + 33) |= 0x40u;
          return v2;
        }
      }
      v7 = 3623879202LL;
      if ( !_bittest64(&v7, v2) )
        goto LABEL_10;
      LODWORD(v2) = sub_23CF1A0((__int64)v6);
      if ( !(_DWORD)v2 )
        goto LABEL_10;
    }
  }
  return v2;
}
