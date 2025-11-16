// Function: sub_2FE5500
// Address: 0x2fe5500
//
__int64 __fastcall sub_2FE5500(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  int v4; // ebx
  unsigned int v5; // r13d
  __int64 v6; // r8
  unsigned __int8 v8; // r14
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // rax
  unsigned int v10; // esi
  int v11; // edx
  int v12; // eax
  __int64 v14; // r9
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int); // rax
  unsigned int v16; // esi
  int v17; // edx
  __int16 v18; // ax
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // r13d
  unsigned int v22; // eax
  __int64 v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // r15d
  __int64 v26; // [rsp+8h] [rbp-48h]
  _QWORD v27[8]; // [rsp+10h] [rbp-40h] BYREF

  v6 = a2;
  v8 = *((_BYTE *)a3 + 8);
  if ( v8 == 14 )
  {
    v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
    v10 = *((_DWORD *)a3 + 2) >> 8;
    if ( v9 == sub_2D42F30 )
    {
      v11 = sub_AE2980(v6, v10)[1];
      LOWORD(v12) = 2;
      if ( v11 != 1 )
      {
        LOWORD(v12) = 3;
        if ( v11 != 2 )
        {
          LOWORD(v12) = 4;
          if ( v11 != 4 )
          {
            LOWORD(v12) = 5;
            if ( v11 != 8 )
            {
              LOWORD(v12) = 6;
              if ( v11 != 16 )
              {
                LOWORD(v12) = 7;
                if ( v11 != 32 )
                {
                  LOWORD(v12) = 8;
                  if ( v11 != 64 )
                    LOWORD(v12) = 9 * (v11 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      LOWORD(v12) = v9(a1, v6, v10);
    }
  }
  else if ( (unsigned int)v8 - 17 > 1 )
  {
    v12 = sub_30097B0(a3, (unsigned __int8)a4, a3, a4, a2);
    HIWORD(v5) = HIWORD(v12);
  }
  else
  {
    v14 = a3[3];
    if ( *(_BYTE *)(v14 + 8) == 14 )
    {
      v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
      v16 = *(_DWORD *)(v14 + 8) >> 8;
      if ( v15 == sub_2D42F30 )
      {
        v17 = sub_AE2980(v6, v16)[1];
        v18 = 2;
        if ( v17 != 1 )
        {
          v18 = 3;
          if ( v17 != 2 )
          {
            v18 = 4;
            if ( v17 != 4 )
            {
              v18 = 5;
              if ( v17 != 8 )
              {
                v18 = 6;
                if ( v17 != 16 )
                {
                  v18 = 7;
                  if ( v17 != 32 )
                  {
                    v18 = 8;
                    if ( v17 != 64 )
                      v18 = 9 * (v17 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v18 = v15(a1, v6, v16);
      }
      v19 = *a3;
      LOWORD(v27[0]) = v18;
      v27[1] = 0;
      v20 = sub_3007410(v27, v19);
      v8 = *((_BYTE *)a3 + 8);
      v14 = v20;
    }
    v21 = *((_DWORD *)a3 + 8);
    v22 = sub_30097B0(v14, 0, a3, a4, v6);
    v23 = *a3;
    v26 = v24;
    v25 = v22;
    LODWORD(v27[0]) = v21;
    BYTE4(v27[0]) = v8 == 18;
    if ( v8 == 18 )
      LOWORD(v12) = sub_2D43AD0(v22, v21);
    else
      LOWORD(v12) = sub_2D43050(v22, v21);
    if ( !(_WORD)v12 )
    {
      v12 = sub_3009450(v23, v25, v26, v27[0]);
      HIWORD(v4) = HIWORD(v12);
    }
    HIWORD(v5) = HIWORD(v4);
  }
  LOWORD(v5) = v12;
  return v5;
}
