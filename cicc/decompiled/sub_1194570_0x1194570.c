// Function: sub_1194570
// Address: 0x1194570
//
__int64 __fastcall sub_1194570(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  unsigned int v4; // eax
  unsigned int v5; // r12d
  __int64 v7; // r12
  _BYTE *v8; // rax
  unsigned int v9; // eax
  int v10; // r13d
  unsigned int v11; // r15d
  _BYTE *v12; // rax
  unsigned int v13; // eax
  __int64 v14; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)sub_BCB060(a2);
  v15 = sub_BCB060(*(_QWORD *)(a1 + 8));
  if ( v15 > 0x40 )
    sub_C43690((__int64)&v14, v2, 0);
  else
    v14 = v2;
  if ( *(_BYTE *)a1 != 17 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    {
      v8 = sub_AD7630(a1, 0, v3);
      if ( v8 && *v8 == 17 )
      {
        LOBYTE(v9) = sub_B532C0((__int64)(v8 + 24), &v14, 32);
        v5 = v9;
        goto LABEL_5;
      }
      if ( *(_BYTE *)(v7 + 8) == 17 )
      {
        v10 = *(_DWORD *)(v7 + 32);
        v5 = 0;
        if ( !v10 )
          goto LABEL_5;
        v11 = 0;
        while ( 1 )
        {
          v12 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a1, v11);
          if ( !v12 )
            break;
          if ( *v12 != 13 )
          {
            if ( *v12 != 17 )
              break;
            LOBYTE(v13) = sub_B532C0((__int64)(v12 + 24), &v14, 32);
            v5 = v13;
            if ( !(_BYTE)v13 )
              break;
          }
          if ( v10 == ++v11 )
            goto LABEL_5;
        }
      }
    }
    v5 = 0;
    goto LABEL_5;
  }
  LOBYTE(v4) = sub_B532C0(a1 + 24, &v14, 32);
  v5 = v4;
LABEL_5:
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return v5;
}
