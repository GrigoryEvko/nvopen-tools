// Function: sub_ED2710
// Address: 0xed2710
//
__int64 __fastcall sub_ED2710(__int64 a1, __int64 a2, int a3, unsigned int a4, _QWORD *a5, unsigned __int8 a6)
{
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned int v19; // edx
  __int64 v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // r11
  __int64 v24; // rsi
  __int64 v25; // rsi
  _QWORD *v26; // r14
  _QWORD *v27; // r11
  unsigned __int64 v28; // rdx
  _QWORD *v29; // rdx
  unsigned int v30; // ecx
  unsigned __int8 v31; // dl
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // r11
  unsigned __int8 v36; // [rsp+Fh] [rbp-51h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  unsigned __int8 v38; // [rsp+10h] [rbp-50h]
  unsigned int v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  const void *v41; // [rsp+20h] [rbp-40h]
  _QWORD *v43; // [rsp+28h] [rbp-38h]
  unsigned int v44; // [rsp+28h] [rbp-38h]

  v41 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v9 = sub_ED2610(a2, a3);
  if ( v9 )
  {
    v10 = *(_BYTE *)(v9 - 16);
    v11 = a6;
    v12 = v9;
    v13 = v9 - 16;
    if ( (v10 & 2) != 0 )
    {
      v14 = *(_DWORD *)(v9 - 24);
      v15 = *(_QWORD *)(v12 - 32);
    }
    else
    {
      v14 = (*(_WORD *)(v9 - 16) >> 6) & 0xF;
      v15 = v13 - 8LL * ((v10 >> 2) & 0xF);
    }
    v16 = *(_QWORD *)(v15 + 16);
    if ( *(_BYTE *)v16 == 1 )
    {
      v17 = *(_QWORD *)(v16 + 136);
      if ( *(_BYTE *)v17 == 17 )
      {
        if ( *(_DWORD *)(v17 + 32) <= 0x40u )
          v18 = *(_QWORD *)(v17 + 24);
        else
          v18 = **(_QWORD **)(v17 + 24);
        *a5 = v18;
        v19 = (v14 - 3) >> 1;
        if ( *(_DWORD *)(a1 + 12) < v19 )
        {
          v38 = a6;
          v40 = v13;
          v44 = v14;
          sub_C8D5F0(a1, v41, v19, 0x10u, v13, v11);
          v11 = v38;
          v13 = v40;
          v14 = v44;
        }
        if ( v14 > 3 )
        {
          v20 = 3;
          while ( 1 )
          {
            v30 = *(_DWORD *)(a1 + 8);
            if ( a4 <= v30 )
              return a1;
            v31 = *(_BYTE *)(v12 - 16);
            v32 = (unsigned int)(v20 + 1);
            if ( (v31 & 2) != 0 )
              break;
            v33 = (v31 >> 2) & 0xF;
            v21 = v13 - 8 * v33;
            v34 = v20 - v33;
            v22 = 0;
            v23 = *(_QWORD *)(v13 + 8 * v34);
            if ( *(_BYTE *)v23 == 1 )
              goto LABEL_13;
            v24 = *(_QWORD *)(v21 + 8 * v32);
            if ( *(_BYTE *)v24 != 1 )
              goto LABEL_32;
LABEL_16:
            v25 = *(_QWORD *)(v24 + 136);
            if ( *(_BYTE *)v25 != 17 || !v22 )
            {
LABEL_32:
              *(_DWORD *)(a1 + 8) = 0;
              return a1;
            }
            v26 = *(_QWORD **)(v25 + 24);
            if ( *(_DWORD *)(v25 + 32) > 0x40u )
              v26 = (_QWORD *)*v26;
            if ( (_BYTE)v11 == 1 || v26 != (_QWORD *)-1LL )
            {
              v27 = *(_QWORD **)(v22 + 24);
              if ( *(_DWORD *)(v22 + 32) > 0x40u )
                v27 = (_QWORD *)*v27;
              v28 = v30 + 1LL;
              if ( v28 > *(unsigned int *)(a1 + 12) )
              {
                v36 = v11;
                v37 = v13;
                v39 = v14;
                v43 = v27;
                sub_C8D5F0(a1, v41, v28, 0x10u, v13, v11);
                v11 = v36;
                v14 = v39;
                v13 = v37;
                v27 = v43;
              }
              v29 = (_QWORD *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
              *v29 = v27;
              v29[1] = v26;
              ++*(_DWORD *)(a1 + 8);
            }
            v20 += 2;
            if ( v14 <= (unsigned int)v20 )
              return a1;
          }
          v21 = *(_QWORD *)(v12 - 32);
          v22 = 0;
          v23 = *(_QWORD *)(v21 + 8 * v20);
          if ( *(_BYTE *)v23 == 1 )
          {
LABEL_13:
            v22 = *(_QWORD *)(v23 + 136);
            if ( *(_BYTE *)v22 != 17 )
              v22 = 0;
          }
          v24 = *(_QWORD *)(v21 + 8 * v32);
          if ( *(_BYTE *)v24 != 1 )
            goto LABEL_32;
          goto LABEL_16;
        }
      }
    }
  }
  return a1;
}
