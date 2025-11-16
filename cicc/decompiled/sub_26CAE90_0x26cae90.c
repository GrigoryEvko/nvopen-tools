// Function: sub_26CAE90
// Address: 0x26cae90
//
_QWORD *__fastcall sub_26CAE90(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // r13
  const char *v7; // r15
  unsigned __int64 v8; // r14
  __int64 v9; // r12
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = sub_B10CD0(a2 + 48);
  if ( v3 )
  {
    v5 = *(_QWORD *)(a2 - 32);
    v6 = v3;
    v7 = 0;
    v8 = 0;
    if ( v5 && !*(_BYTE *)v5 && *(_QWORD *)(v5 + 24) == *(_QWORD *)(a2 + 80) )
    {
      v7 = sub_BD5D20(v5);
      v8 = v4;
      if ( !unk_4F838D3 )
      {
LABEL_6:
        v9 = sub_26CAC90((__int64)a1, a2, v4);
        if ( v9 )
        {
          v11 = *(_QWORD *)(a1[142] + 88LL);
          v12[0] = sub_C1B090(v6, 0);
          return sub_C1BBC0(v9, (__int64)v12, (unsigned __int64)v7, v8, v11, a1 + 169);
        }
        return 0;
      }
    }
    else if ( !unk_4F838D3 )
    {
      goto LABEL_6;
    }
    return (_QWORD *)sub_317ED60(a1[189], a2, v7, v8);
  }
  return 0;
}
