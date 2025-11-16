// Function: sub_1F04120
// Address: 0x1f04120
//
unsigned __int16 __fastcall sub_1F04120(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  _QWORD *v5; // rdx
  unsigned int v6; // edi
  __int64 v7; // rax
  __int64 v9; // r14
  __int64 v10; // rsi
  unsigned int v11; // ecx
  _WORD *v12; // rcx
  unsigned __int16 result; // ax
  _WORD *v14; // rsi
  int v15; // edi
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rcx
  unsigned int v19; // r8d
  unsigned int v20; // edx
  __int64 v21; // rcx
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rsi
  int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // rcx
  unsigned __int64 v31; // r9
  void (*v32)(); // rax
  __int64 v33; // rax
  unsigned int v34; // edx
  __int64 v35; // rbx
  __int64 v36; // r15
  __int64 v37; // r8
  unsigned __int64 v39; // [rsp+20h] [rbp-80h] BYREF
  __int64 v40; // [rsp+28h] [rbp-78h]
  unsigned int v41; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v42; // [rsp+38h] [rbp-68h]
  char v43; // [rsp+40h] [rbp-60h]
  unsigned __int16 v44; // [rsp+48h] [rbp-58h]
  _WORD *v45; // [rsp+50h] [rbp-50h]
  int v46; // [rsp+58h] [rbp-48h]
  unsigned __int16 v47; // [rsp+60h] [rbp-40h]
  __int64 v48; // [rsp+68h] [rbp-38h]

  v4 = 5LL * a3;
  v5 = *(_QWORD **)(a1 + 24);
  v6 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 8) + 32LL) + 8 * v4 + 8);
  if ( !v5 )
  {
    v41 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 8) + 32LL) + 8 * v4 + 8);
    v42 = 0;
    v43 = 1;
    v44 = 0;
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    BUG();
  }
  v7 = *(_QWORD *)(a1 + 32);
  v41 = v6;
  v9 = *(_QWORD *)(v7 + 16);
  v43 = 1;
  v44 = 0;
  v45 = 0;
  v48 = 0;
  v10 = v5[7];
  v42 = v5 + 1;
  v46 = 0;
  v47 = 0;
  v11 = *(_DWORD *)(v5[1] + 24LL * v6 + 16);
  LOWORD(v7) = v6 * (v11 & 0xF);
  v12 = (_WORD *)(v10 + 2LL * (v11 >> 4));
  result = *v12 + v7;
  v14 = v12 + 1;
  v44 = result;
  v45 = v12 + 1;
  while ( v14 )
  {
    v46 = *(_DWORD *)(v5[6] + 4LL * v44);
    v15 = (unsigned __int16)v46;
    if ( (_WORD)v46 )
    {
      while ( 1 )
      {
        v16 = (unsigned __int16)v15;
        v17 = *(unsigned int *)(v5[1] + 24LL * (unsigned __int16)v15 + 8);
        v18 = v5[7];
        v47 = v15;
        v48 = v18 + 2 * v17;
        if ( v48 )
          break;
        v46 = HIWORD(v46);
        v15 = v46;
        if ( !(_WORD)v46 )
          goto LABEL_32;
      }
      while ( 1 )
      {
        v19 = *(_DWORD *)(a1 + 1224);
        v20 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 1424) + 2 * v16);
        if ( v20 < v19 )
        {
          v21 = *(_QWORD *)(a1 + 1216);
          v22 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 1424) + 2 * v16);
          while ( 1 )
          {
            v23 = v21 + 24LL * v22;
            if ( v15 == *(_DWORD *)(v23 + 12) )
            {
              v24 = *(unsigned int *)(v23 + 16);
              if ( (_DWORD)v24 != -1 && *(_DWORD *)(v21 + 24 * v24 + 20) == -1 )
                break;
            }
            v22 += 0x10000;
            if ( v19 <= v22 )
              goto LABEL_29;
          }
          if ( v22 != -1 )
          {
            while ( 1 )
            {
              v25 = v20;
              v26 = v21 + 24LL * v20;
              if ( v15 == *(_DWORD *)(v26 + 12) )
              {
                v27 = *(unsigned int *)(v26 + 16);
                if ( (_DWORD)v27 != -1 && *(_DWORD *)(v21 + 24 * v27 + 20) == -1 )
                  break;
              }
              v20 += 0x10000;
              if ( v19 <= v20 )
                goto LABEL_29;
            }
            if ( v20 != -1 )
            {
              while ( 1 )
              {
                v35 = 24 * v25;
                v33 = v21 + 24 * v25;
                v36 = *(_QWORD *)v33;
                if ( a2 != *(_QWORD *)v33 )
                {
                  v37 = *(unsigned int *)(v33 + 8);
                  if ( (int)v37 >= 0 )
                  {
                    v28 = v47;
                    HIDWORD(v40) = 1;
                    *(_BYTE *)(a2 + 228) |= 0x40u;
                    v39 = a2 & 0xFFFFFFFFFFFFFFF9LL;
                    LODWORD(v40) = v28;
                    v29 = *(_QWORD *)(v36 + 8);
                  }
                  else
                  {
                    v40 = 3;
                    v29 = 0;
                    v39 = a2 & 0xFFFFFFFFFFFFFFF9LL | 6;
                  }
                  HIDWORD(v40) = sub_1F4BB70(a1 + 632, *(_QWORD *)(a2 + 8), a3, v29, v37);
                  v32 = *(void (**)())(*(_QWORD *)v9 + 208LL);
                  if ( v32 != nullsub_681 )
                    ((void (__fastcall *)(__int64, __int64, __int64, unsigned __int64 *))v32)(v9, a2, v36, &v39);
                  sub_1F01A00(v36, (__int64)&v39, 1, v30, (int)&v39, v31);
                  v21 = *(_QWORD *)(a1 + 1216);
                  v33 = v21 + v35;
                }
                v34 = *(_DWORD *)(v33 + 20);
                if ( v34 == -1 )
                  break;
                v25 = v34;
              }
            }
          }
        }
LABEL_29:
        result = sub_1E1D5E0((__int64)&v41);
        if ( !v45 )
          return result;
        v15 = v47;
        v16 = v47;
      }
    }
LABEL_32:
    v45 = ++v14;
    result = *(v14 - 1);
    v44 += result;
    if ( !result )
      return result;
  }
  return result;
}
