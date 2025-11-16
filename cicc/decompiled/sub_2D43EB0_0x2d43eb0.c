// Function: sub_2D43EB0
// Address: 0x2d43eb0
//
__int64 __fastcall sub_2D43EB0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  unsigned __int8 v6; // r14
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int); // rax
  unsigned int v8; // esi
  int v9; // eax
  __int64 v10; // r9
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // rax
  unsigned int v12; // esi
  int v13; // edx
  __int16 v14; // ax
  __int64 v15; // rsi
  __int64 v16; // rax
  int v17; // r12d
  unsigned int v18; // eax
  __int64 v19; // r13
  __int64 v20; // rdx
  unsigned int v21; // r15d
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // eax
  int v27; // edx
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int16 v29; // [rsp+10h] [rbp-60h] BYREF
  __int64 v30; // [rsp+18h] [rbp-58h]
  unsigned __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  char v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h] BYREF
  __int64 v34; // [rsp+38h] [rbp-38h]

  v4 = a1;
  v6 = *((_BYTE *)a2 + 8);
  if ( v6 != 14 )
  {
    if ( (unsigned int)v6 - 17 > 1 )
    {
      LOWORD(v9) = sub_30097B0(a2, 0, a3, a4, a1);
    }
    else
    {
      v10 = a2[3];
      if ( *(_BYTE *)(v10 + 8) == 14 )
      {
        v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 40LL);
        v12 = *(_DWORD *)(v10 + 8) >> 8;
        if ( v11 == sub_2D42FA0 )
        {
          v13 = sub_AE2980(a3, v12)[1];
          v14 = 2;
          if ( v13 != 1 )
          {
            v14 = 3;
            if ( v13 != 2 )
            {
              v14 = 4;
              if ( v13 != 4 )
              {
                v14 = 5;
                if ( v13 != 8 )
                {
                  v14 = 6;
                  if ( v13 != 16 )
                  {
                    v14 = 7;
                    if ( v13 != 32 )
                    {
                      v14 = 8;
                      if ( v13 != 64 )
                        v14 = 9 * (v13 == 128);
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v14 = v11(a1, a3, v12);
        }
        v15 = *a2;
        LOWORD(v33) = v14;
        v34 = 0;
        v16 = sub_3007410(&v33, v15);
        v6 = *((_BYTE *)a2 + 8);
        v10 = v16;
      }
      v17 = *((_DWORD *)a2 + 8);
      v18 = sub_30097B0(v10, 0, a3, a4, v4);
      v19 = *a2;
      v28 = v20;
      v21 = v18;
      LODWORD(v33) = v17;
      BYTE4(v33) = v6 == 18;
      if ( v6 == 18 )
        LOWORD(v9) = sub_2D43AD0(v18, v17);
      else
        LOWORD(v9) = sub_2D43050(v18, v17);
      v22 = 0;
      if ( !(_WORD)v9 )
        LOWORD(v9) = sub_3009450(v19, v21, v28, v33);
    }
LABEL_28:
    v29 = v9;
    v30 = v22;
    if ( !(_WORD)v9 )
      goto LABEL_29;
    v27 = (unsigned __int16)v9;
    if ( (_WORD)v9 == 1 )
      goto LABEL_49;
    goto LABEL_38;
  }
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 40LL);
  v8 = *((_DWORD *)a2 + 2) >> 8;
  if ( v7 != sub_2D42FA0 )
  {
    LOWORD(v9) = v7(a1, a3, v8);
    v22 = 0;
    goto LABEL_28;
  }
  v9 = sub_AE2980(a3, v8)[1];
  switch ( v9 )
  {
    case 1:
      v30 = 0;
      LOWORD(v9) = 2;
      v29 = 2;
LABEL_37:
      v27 = (unsigned __int16)v9;
LABEL_38:
      if ( (unsigned __int16)(v9 - 504) > 7u )
        goto LABEL_34;
LABEL_49:
      BUG();
    case 2:
      v30 = 0;
      v27 = 3;
      v29 = 3;
LABEL_34:
      v24 = 16LL * (v27 - 1) + 71615648;
      v23 = *(_QWORD *)v24;
      LOBYTE(v24) = *(_BYTE *)(v24 + 8);
      goto LABEL_30;
    case 4:
      v30 = 0;
      v29 = 4;
      goto LABEL_37;
    case 8:
      v30 = 0;
      v27 = 5;
      v29 = 5;
      goto LABEL_34;
    case 16:
      v30 = 0;
      LOWORD(v9) = 6;
      v29 = 6;
      goto LABEL_37;
    case 32:
      v30 = 0;
      v27 = 7;
      v29 = 7;
      goto LABEL_34;
    case 64:
      v30 = 0;
      LOWORD(v9) = 8;
      v29 = 8;
      goto LABEL_37;
    case 128:
      v30 = 0;
      v27 = 9;
      v29 = 9;
      goto LABEL_34;
  }
  v30 = 0;
  v29 = 0;
LABEL_29:
  v23 = sub_3007260(&v29);
  v33 = v23;
  v34 = v24;
LABEL_30:
  v32 = v24;
  v31 = (v23 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v25 = sub_CA1930(&v31);
  return sub_BCCE00((_QWORD *)*a2, v25);
}
