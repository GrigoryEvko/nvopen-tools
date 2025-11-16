// Function: sub_38AC1E0
// Address: 0x38ac1e0
//
__int64 __fastcall sub_38AC1E0(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned int v7; // r15d
  __int64 v9; // rax
  __int64 v10; // r12
  __int16 v11; // ax
  __int16 v12; // dx
  __int64 v13; // r8
  int v14; // eax
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // rsi
  const char *v21; // rax
  char v22; // al
  __int64 v23; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+0h] [rbp-80h]
  __int64 v25; // [rsp+0h] [rbp-80h]
  unsigned __int64 v26; // [rsp+8h] [rbp-78h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-60h] BYREF
  __int64 v30; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v31[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v32; // [rsp+40h] [rbp-40h]

  v29 = 0;
  v31[0] = "expected type";
  v32 = 259;
  v7 = sub_3891B00(a1, &v29, (__int64)v31, 0);
  if ( !(_BYTE)v7 )
  {
    v32 = 257;
    v9 = sub_15F5910(v29, 0, (__int64)v31, 0);
    v10 = v9;
    if ( *(_DWORD *)(a1 + 64) == 272 )
    {
      *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
      v11 = *(_WORD *)(v10 + 18);
      v12 = v11 & 0x7FFE | 1;
    }
    else
    {
      v11 = *(_WORD *)(v9 + 18);
      v12 = v11 & 0x7FFE;
    }
    v13 = a1 + 8;
    *(_WORD *)(v10 + 18) = v12 | v11 & 0x8000;
    v14 = *(_DWORD *)(a1 + 64);
    if ( v14 == 273 )
      goto LABEL_13;
LABEL_6:
    if ( v14 == 274 )
    {
      v23 = v13;
      *(_DWORD *)(a1 + 64) = sub_3887100(v13);
      v26 = *(_QWORD *)(a1 + 56);
      v15 = sub_38AB270((__int64 **)a1, &v30, a3, a4, a5, a6);
      v19 = v23;
      if ( v15 )
      {
LABEL_19:
        v7 = 1;
      }
      else
      {
        v20 = v30;
        if ( *(_BYTE *)(*(_QWORD *)v30 + 8LL) == 14 )
          goto LABEL_11;
        HIBYTE(v32) = 1;
        v21 = "'filter' clause has an invalid type";
LABEL_10:
        v24 = v19;
        v31[0] = v21;
        LOBYTE(v32) = 3;
        sub_38814C0(v19, v26, (__int64)v31);
        v20 = v30;
        v19 = v24;
LABEL_11:
        while ( *(_BYTE *)(v20 + 16) <= 0x10u )
        {
          v27 = v19;
          sub_15F5A60(v10, v20, v16, v17, v19, v18);
          v14 = *(_DWORD *)(a1 + 64);
          v13 = v27;
          if ( v14 != 273 )
            goto LABEL_6;
LABEL_13:
          v25 = v13;
          *(_DWORD *)(a1 + 64) = sub_3887100(v13);
          v26 = *(_QWORD *)(a1 + 56);
          v22 = sub_38AB270((__int64 **)a1, &v30, a3, a4, a5, a6);
          v19 = v25;
          if ( v22 )
            goto LABEL_19;
          v20 = v30;
          if ( *(_BYTE *)(*(_QWORD *)v30 + 8LL) == 14 )
          {
            HIBYTE(v32) = 1;
            v21 = "'catch' clause has an invalid type";
            goto LABEL_10;
          }
        }
        v32 = 259;
        v31[0] = "clause argument must be a constant";
        v7 = sub_38814C0(v19, v26, (__int64)v31);
      }
      sub_15F2000(v10);
      sub_1648B90(v10);
    }
    else
    {
      *a2 = v10;
    }
  }
  return v7;
}
