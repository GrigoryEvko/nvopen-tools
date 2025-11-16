// Function: sub_2D5BAE0
// Address: 0x2d5bae0
//
__int64 __fastcall sub_2D5BAE0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r8
  unsigned __int8 v8; // r13
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // rax
  unsigned int v10; // esi
  _DWORD *v11; // rax
  unsigned __int16 v12; // dx
  int v13; // eax
  __int64 v15; // r9
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int); // rax
  unsigned int v17; // esi
  int v18; // edx
  __int16 v19; // ax
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // r14d
  unsigned int v23; // eax
  __int64 v24; // r12
  __int64 v25; // rdx
  unsigned int v26; // r15d
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-48h]
  _QWORD v29[8]; // [rsp+10h] [rbp-40h] BYREF

  v5 = a1;
  v8 = *((_BYTE *)a3 + 8);
  if ( v8 == 14 )
  {
    v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v5 + 32LL);
    v10 = *((_DWORD *)a3 + 2) >> 8;
    if ( v9 == sub_2D42F30 )
    {
      v11 = sub_AE2980(a2, v10);
      v12 = 2;
      v13 = v11[1];
      if ( v13 != 1 )
      {
        v12 = 3;
        if ( v13 != 2 )
        {
          v12 = 4;
          if ( v13 != 4 )
          {
            v12 = 5;
            if ( v13 != 8 )
            {
              v12 = 6;
              if ( v13 != 16 )
              {
                v12 = 7;
                if ( v13 != 32 )
                {
                  v12 = 8;
                  if ( v13 != 64 )
                    return (unsigned __int16)(9 * (v13 == 128));
                }
              }
            }
          }
        }
      }
    }
    else
    {
      return (unsigned __int16)v9(v5, a2, v10);
    }
    return v12;
  }
  else if ( (unsigned int)v8 - 17 > 1 )
  {
    return sub_30097B0(a3, (unsigned __int8)a4, a3, a4, v5);
  }
  else
  {
    v15 = a3[3];
    if ( *(_BYTE *)(v15 + 8) == 14 )
    {
      v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v5 + 32LL);
      v17 = *(_DWORD *)(v15 + 8) >> 8;
      if ( v16 == sub_2D42F30 )
      {
        v18 = sub_AE2980(a2, v17)[1];
        v19 = 2;
        if ( v18 != 1 )
        {
          v19 = 3;
          if ( v18 != 2 )
          {
            v19 = 4;
            if ( v18 != 4 )
            {
              v19 = 5;
              if ( v18 != 8 )
              {
                v19 = 6;
                if ( v18 != 16 )
                {
                  v19 = 7;
                  if ( v18 != 32 )
                  {
                    v19 = 8;
                    if ( v18 != 64 )
                      v19 = 9 * (v18 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v19 = v16(v5, a2, v17);
      }
      v20 = *a3;
      LOWORD(v29[0]) = v19;
      v29[1] = 0;
      v21 = sub_3007410(v29, v20);
      v8 = *((_BYTE *)a3 + 8);
      v15 = v21;
    }
    v22 = *((_DWORD *)a3 + 8);
    v23 = sub_30097B0(v15, 0, a3, a4, v5);
    v24 = *a3;
    v28 = v25;
    v26 = v23;
    LODWORD(v29[0]) = v22;
    BYTE4(v29[0]) = v8 == 18;
    if ( v8 == 18 )
      LOWORD(v27) = sub_2D43AD0(v23, v22);
    else
      LOWORD(v27) = sub_2D43050(v23, v22);
    if ( !(_WORD)v27 )
    {
      v27 = sub_3009450(v24, v26, v28, v29[0]);
      v4 = v27;
    }
    LOWORD(v4) = v27;
    return v4;
  }
}
