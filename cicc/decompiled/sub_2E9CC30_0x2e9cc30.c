// Function: sub_2E9CC30
// Address: 0x2e9cc30
//
__int64 __fastcall sub_2E9CC30(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  _DWORD *v5; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v10; // r14
  int v11; // eax
  char v12; // bl
  int *v13; // rax
  unsigned __int64 v14; // r9
  int v15; // r8d
  char v16; // al
  int *v18; // rax
  int v19; // ebx
  int *v20; // r14
  __int64 v21; // r11
  int v22; // r13d
  __int64 v23; // r9
  int v24; // esi
  unsigned int v25; // ecx
  _DWORD *v26; // rdi
  int v27; // r10d
  char v28; // dl
  unsigned int v29; // esi
  unsigned int v30; // ecx
  _DWORD *v31; // rax
  int v32; // edi
  unsigned int v33; // r8d
  int *v34; // rax
  int v35; // r8d
  int v36; // edx
  __int64 v37; // rsi
  int v38; // edx
  unsigned int v39; // ecx
  int v40; // edi
  __int64 v41; // rsi
  int v42; // edx
  unsigned int v43; // ecx
  int v44; // edi
  int v45; // r10d
  _DWORD *v46; // r8
  char v47; // al
  int v48; // edx
  int v49; // r10d
  __int64 v50; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v51; // [rsp+0h] [rbp-A0h]
  __int64 v52; // [rsp+8h] [rbp-98h]
  unsigned __int64 v55; // [rsp+18h] [rbp-88h]
  int v56; // [rsp+18h] [rbp-88h]
  __int64 v57; // [rsp+18h] [rbp-88h]
  __int64 v58; // [rsp+18h] [rbp-88h]
  int v59; // [rsp+18h] [rbp-88h]
  __int64 v61; // [rsp+28h] [rbp-78h]
  int v62; // [rsp+3Ch] [rbp-64h] BYREF
  _BYTE v63[96]; // [rsp+40h] [rbp-60h] BYREF

  v5 = (_DWORD *)(a1 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 1;
  v52 = a1 + 16;
  do
  {
    if ( v5 )
      *v5 = -1;
    v5 += 2;
  }
  while ( v5 != (_DWORD *)(a1 + 48) );
  if ( *(_WORD *)(a3 + 68) != 10 )
  {
    v7 = *(unsigned __int16 *)(*(_QWORD *)(a3 + 16) + 2LL);
    if ( (_WORD)v7 )
    {
      v8 = 0;
      v61 = 40 * v7;
      do
      {
        v10 = v8 + *(_QWORD *)(a3 + 32);
        if ( *(_BYTE *)v10 )
          goto LABEL_15;
        if ( (*(_BYTE *)(v10 + 3) & 0x20) != 0 )
          goto LABEL_15;
        v11 = *(_DWORD *)(v10 + 8);
        v62 = v11;
        if ( v11 >= 0 )
          goto LABEL_15;
        v12 = 0;
        if ( a4 )
        {
          sub_2E9C8D0((__int64)v63, a2 + 512, &v62);
          v12 = v63[32];
          v11 = v62;
        }
        v55 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 56LL) + 16LL * (v11 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
        v13 = (int *)(*(__int64 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a2 + 16) + 376LL))(
                       *(_QWORD *)(a2 + 16),
                       v55);
        v14 = v55;
        v15 = *v13;
        v16 = *(_BYTE *)(v10 + 3);
        if ( (v16 & 0x10) == 0 )
        {
          if ( (v16 & 0x40) != 0
            || (v51 = v55,
                v59 = v15,
                v47 = sub_2EBEF70(*(_QWORD *)(a2 + 32), *(unsigned int *)(v10 + 8)),
                v15 = v59,
                v14 = v51,
                v47) )
          {
            if ( v12 )
              goto LABEL_15;
            v15 = -v15;
          }
          else if ( !v12 || !a5 )
          {
            goto LABEL_15;
          }
        }
        v56 = v15;
        if ( v15 )
        {
          v18 = (int *)(*(__int64 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a2 + 16) + 416LL))(
                         *(_QWORD *)(a2 + 16),
                         v14);
          v19 = *v18;
          v20 = v18;
          if ( *v18 != -1 )
          {
            v50 = v8;
            v21 = v52;
            v22 = v56;
            while ( 1 )
            {
              v28 = *(_BYTE *)(a1 + 8) & 1;
              if ( v28 )
              {
                v23 = v21;
                v24 = 3;
              }
              else
              {
                v29 = *(_DWORD *)(a1 + 24);
                v23 = *(_QWORD *)(a1 + 16);
                if ( !v29 )
                {
                  v30 = *(_DWORD *)(a1 + 8);
                  ++*(_QWORD *)a1;
                  v31 = 0;
                  v32 = (v30 >> 1) + 1;
LABEL_26:
                  v33 = 3 * v29;
                  goto LABEL_27;
                }
                v24 = v29 - 1;
              }
              v25 = v24 & (37 * v19);
              v26 = (_DWORD *)(v23 + 8LL * v25);
              v27 = *v26;
              if ( v19 == *v26 )
              {
LABEL_21:
                ++v20;
                v26[1] += v22;
                v19 = *v20;
                if ( *v20 == -1 )
                  goto LABEL_32;
              }
              else
              {
                v35 = 1;
                v31 = 0;
                while ( v27 != -1 )
                {
                  if ( v27 == -2 && !v31 )
                    v31 = v26;
                  v25 = v24 & (v35 + v25);
                  v26 = (_DWORD *)(v23 + 8LL * v25);
                  v27 = *v26;
                  if ( v19 == *v26 )
                    goto LABEL_21;
                  ++v35;
                }
                v30 = *(_DWORD *)(a1 + 8);
                v33 = 12;
                v29 = 4;
                if ( !v31 )
                  v31 = v26;
                ++*(_QWORD *)a1;
                v32 = (v30 >> 1) + 1;
                if ( !v28 )
                {
                  v29 = *(_DWORD *)(a1 + 24);
                  goto LABEL_26;
                }
LABEL_27:
                if ( 4 * v32 >= v33 )
                {
                  v57 = v21;
                  sub_2E9A4F0(a1, 2 * v29);
                  v21 = v57;
                  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
                  {
                    v37 = v57;
                    v38 = 3;
                  }
                  else
                  {
                    v36 = *(_DWORD *)(a1 + 24);
                    v37 = *(_QWORD *)(a1 + 16);
                    if ( !v36 )
                      goto LABEL_78;
                    v38 = v36 - 1;
                  }
                  v39 = v38 & (37 * v19);
                  v31 = (_DWORD *)(v37 + 8LL * v39);
                  v40 = *v31;
                  if ( v19 != *v31 )
                  {
                    v49 = 1;
                    v46 = 0;
                    while ( v40 != -1 )
                    {
                      if ( !v46 && v40 == -2 )
                        v46 = v31;
                      v39 = v38 & (v39 + v49);
                      v31 = (_DWORD *)(v37 + 8LL * v39);
                      v40 = *v31;
                      if ( v19 == *v31 )
                        goto LABEL_46;
                      ++v49;
                    }
LABEL_53:
                    if ( v46 )
                      v31 = v46;
                  }
LABEL_46:
                  v30 = *(_DWORD *)(a1 + 8);
                  goto LABEL_29;
                }
                if ( v29 - *(_DWORD *)(a1 + 12) - v32 <= v29 >> 3 )
                {
                  v58 = v21;
                  sub_2E9A4F0(a1, v29);
                  v21 = v58;
                  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
                  {
                    v41 = v58;
                    v42 = 3;
                  }
                  else
                  {
                    v48 = *(_DWORD *)(a1 + 24);
                    v41 = *(_QWORD *)(a1 + 16);
                    if ( !v48 )
                    {
LABEL_78:
                      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
                      BUG();
                    }
                    v42 = v48 - 1;
                  }
                  v43 = v42 & (37 * v19);
                  v31 = (_DWORD *)(v41 + 8LL * v43);
                  v44 = *v31;
                  if ( v19 != *v31 )
                  {
                    v45 = 1;
                    v46 = 0;
                    while ( v44 != -1 )
                    {
                      if ( v44 == -2 && !v46 )
                        v46 = v31;
                      v43 = v42 & (v43 + v45);
                      v31 = (_DWORD *)(v41 + 8LL * v43);
                      v44 = *v31;
                      if ( v19 == *v31 )
                        goto LABEL_46;
                      ++v45;
                    }
                    goto LABEL_53;
                  }
                  goto LABEL_46;
                }
LABEL_29:
                *(_DWORD *)(a1 + 8) = (2 * (v30 >> 1) + 2) | v30 & 1;
                if ( *v31 != -1 )
                  --*(_DWORD *)(a1 + 12);
                *v31 = v19;
                v34 = v31 + 1;
                ++v20;
                *v34 = 0;
                *v34 = v22;
                v19 = *v20;
                if ( *v20 == -1 )
                {
LABEL_32:
                  v8 = v50;
                  break;
                }
              }
            }
          }
        }
LABEL_15:
        v8 += 40;
      }
      while ( v8 != v61 );
    }
  }
  return a1;
}
