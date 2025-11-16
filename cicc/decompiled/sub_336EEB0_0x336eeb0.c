// Function: sub_336EEB0
// Address: 0x336eeb0
//
__int64 __fastcall sub_336EEB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r8
  unsigned __int8 v7; // r13
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int); // rax
  unsigned int v9; // esi
  _DWORD *v10; // rax
  unsigned __int16 v11; // dx
  int v12; // eax
  __int64 v14; // r9
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int); // rax
  unsigned int v16; // esi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int16 v21; // ax
  __int64 *v22; // rsi
  __int64 v23; // rax
  int v24; // r14d
  unsigned int v25; // eax
  __int64 *v26; // r12
  __int64 v27; // rdx
  unsigned int v28; // r15d
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33[8]; // [rsp+10h] [rbp-40h] BYREF

  v5 = a2;
  v7 = *(_BYTE *)(a3 + 8);
  if ( v7 == 14 )
  {
    v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 40LL);
    v9 = *(_DWORD *)(a3 + 8) >> 8;
    if ( v8 == sub_2D42FA0 )
    {
      v10 = sub_AE2980(v5, v9);
      v11 = 2;
      v12 = v10[1];
      if ( v12 != 1 )
      {
        v11 = 3;
        if ( v12 != 2 )
        {
          v11 = 4;
          if ( v12 != 4 )
          {
            v11 = 5;
            if ( v12 != 8 )
            {
              v11 = 6;
              if ( v12 != 16 )
              {
                v11 = 7;
                if ( v12 != 32 )
                {
                  v11 = 8;
                  if ( v12 != 64 )
                    return (unsigned __int16)(9 * (v12 == 128));
                }
              }
            }
          }
        }
      }
    }
    else
    {
      return (unsigned __int16)v8(a1, v5, v9);
    }
    return v11;
  }
  else if ( (unsigned int)v7 - 17 > 1 )
  {
    return sub_2D5BAE0(a1, a2, (__int64 *)a3, (unsigned __int8)a4);
  }
  else
  {
    v14 = *(_QWORD *)(a3 + 24);
    if ( *(_BYTE *)(v14 + 8) == 14 )
    {
      v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 40LL);
      v16 = *(_DWORD *)(v14 + 8) >> 8;
      if ( v15 == sub_2D42FA0 )
      {
        v17 = (unsigned int)sub_AE2980(v5, v16)[1];
        v21 = 2;
        if ( (_DWORD)v17 != 1 )
        {
          v21 = 3;
          if ( (_DWORD)v17 != 2 )
          {
            v21 = 4;
            if ( (_DWORD)v17 != 4 )
            {
              v21 = 5;
              if ( (_DWORD)v17 != 8 )
              {
                v21 = 6;
                if ( (_DWORD)v17 != 16 )
                {
                  v21 = 7;
                  if ( (_DWORD)v17 != 32 )
                  {
                    v21 = 8;
                    if ( (_DWORD)v17 != 64 )
                      v21 = 9 * ((_DWORD)v17 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v21 = v15(a1, v5, v16);
      }
      v22 = *(__int64 **)a3;
      LOWORD(v33[0]) = v21;
      v33[1] = 0;
      v23 = sub_3007410((__int64)v33, v22, v17, v18, v19, v20);
      v7 = *(_BYTE *)(a3 + 8);
      v14 = v23;
    }
    v24 = *(_DWORD *)(a3 + 32);
    v25 = sub_30097B0(v14, 0, a3, a4, v5);
    v26 = *(__int64 **)a3;
    v32 = v27;
    v28 = v25;
    LODWORD(v33[0]) = v24;
    BYTE4(v33[0]) = v7 == 18;
    if ( v7 == 18 )
      LOWORD(v29) = sub_2D43AD0(v25, v24);
    else
      LOWORD(v29) = sub_2D43050(v25, v24);
    if ( !(_WORD)v29 )
    {
      v29 = sub_3009450(v26, v28, v32, v33[0], v30, v31);
      v4 = v29;
    }
    LOWORD(v4) = v29;
    return v4;
  }
}
