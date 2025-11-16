// Function: sub_D4B180
// Address: 0xd4b180
//
__int64 __fastcall sub_D4B180(__int64 a1, unsigned __int8 *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // r13
  _QWORD *v9; // r12
  unsigned int v10; // r10d
  char v12; // al
  char v13; // al
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int8 *v17; // r11
  __int64 v18; // r12
  unsigned __int8 *v19; // r15
  unsigned int v20; // eax
  int v21; // ecx
  __int64 v22; // rsi
  int v23; // ecx
  unsigned int v24; // edx
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rax
  int v29; // eax
  int v30; // r8d
  unsigned __int8 v31; // [rsp+Fh] [rbp-41h]
  unsigned __int8 *v32; // [rsp+10h] [rbp-40h]

  v8 = a4;
  v9 = a2;
  v10 = sub_D48480(a1, (__int64)a2, (__int64)a3, a4);
  if ( !(_BYTE)v10 )
  {
    v12 = sub_991A70(a2, 0, 0, 0, 0, 1u, 0);
    v10 = 0;
    v31 = v12;
    if ( v12 )
    {
      v13 = sub_B46420((__int64)a2);
      v10 = 0;
      if ( !v13 )
      {
        v14 = (unsigned int)*a2 - 39;
        if ( (unsigned int)v14 > 0x38 || (v15 = 0x100060000000001LL, !_bittest64(&v15, v14)) )
        {
          if ( !v8 )
          {
            v28 = sub_D4B130(a1);
            v10 = 0;
            if ( !v28 )
              return v10;
            v8 = sub_986580(v28);
          }
          v16 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
          if ( (a2[7] & 0x40) != 0 )
          {
            v17 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
            v32 = &v17[v16];
          }
          else
          {
            v32 = a2;
            v17 = &a2[-v16];
          }
          if ( v32 != v17 )
          {
            v18 = a6;
            v19 = v17;
            do
            {
              v20 = sub_D4B3B0(a1, *(_QWORD *)v19, a3, v8, a5, v18);
              if ( !(_BYTE)v20 )
                return v20;
              v19 += 32;
            }
            while ( v32 != v19 );
            a6 = v18;
            v9 = a2;
          }
          sub_B444E0(v9, v8 + 24, 0);
          if ( a5 )
          {
            v21 = *(_DWORD *)(*(_QWORD *)a5 + 56LL);
            v22 = *(_QWORD *)(*(_QWORD *)a5 + 40LL);
            if ( v21 )
            {
              v23 = v21 - 1;
              v24 = v23 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
              v25 = (_QWORD *)(v22 + 16LL * v24);
              v26 = (_QWORD *)*v25;
              if ( v9 == (_QWORD *)*v25 )
              {
LABEL_19:
                v27 = v25[1];
                if ( v27 )
                  sub_D75590(a5, v27, *(_QWORD *)(v8 + 40), 2);
              }
              else
              {
                v29 = 1;
                while ( v26 != (_QWORD *)-4096LL )
                {
                  v30 = v29 + 1;
                  v24 = v23 & (v29 + v24);
                  v25 = (_QWORD *)(v22 + 16LL * v24);
                  v26 = (_QWORD *)*v25;
                  if ( v9 == (_QWORD *)*v25 )
                    goto LABEL_19;
                  v29 = v30;
                }
              }
            }
          }
          sub_B9ADA0((__int64)v9, 0, 0);
          if ( a6 )
            sub_D9D700(a6, v9);
          v10 = v31;
          *a3 = 1;
        }
      }
    }
  }
  return v10;
}
