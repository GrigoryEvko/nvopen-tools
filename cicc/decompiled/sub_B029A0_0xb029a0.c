// Function: sub_B029A0
// Address: 0xb029a0
//
__int64 __fastcall sub_B029A0(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6, char a7)
{
  unsigned int v8; // r13d
  __int64 *v9; // r12
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // rcx
  int v13; // r15d
  unsigned int i; // ebx
  __int64 *v15; // r12
  __int64 v16; // r13
  int v17; // r15d
  unsigned int v18; // r8d
  int v19; // r15d
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rdi
  __int64 result; // rax
  __int64 *v24; // r15
  __int64 v25; // rax
  int v26; // eax
  _QWORD *v27; // r15
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  int v31; // eax
  _QWORD *v32; // r15
  __int64 v33; // rsi
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // [rsp+10h] [rbp-B0h]
  int v37; // [rsp+1Ch] [rbp-A4h]
  int v38; // [rsp+20h] [rbp-A0h]
  unsigned int v39; // [rsp+24h] [rbp-9Ch]
  __int64 v40; // [rsp+28h] [rbp-98h]
  __int64 v41; // [rsp+38h] [rbp-88h]
  __int16 v43; // [rsp+4Ch] [rbp-74h]
  int v44; // [rsp+5Ch] [rbp-64h] BYREF
  _QWORD *v45; // [rsp+60h] [rbp-60h] BYREF
  __int64 v46; // [rsp+68h] [rbp-58h]
  _QWORD *v47; // [rsp+70h] [rbp-50h]
  __int64 v48; // [rsp+78h] [rbp-48h]
  int v49; // [rsp+80h] [rbp-40h]
  int v50; // [rsp+84h] [rbp-3Ch] BYREF
  __int64 v51[7]; // [rsp+88h] [rbp-38h] BYREF

  v8 = a6;
  v9 = a1;
  v10 = a5;
  v43 = a2;
  if ( a6 )
  {
    v19 = 0;
    goto LABEL_11;
  }
  v45 = (_QWORD *)a4;
  v46 = a5;
  v47 = 0;
  v48 = 0;
  v11 = sub_B75C00(a4, a5);
  v12 = *a1;
  v50 = a2;
  v49 = v11;
  v51[0] = a3;
  v13 = *(_DWORD *)(v12 + 816);
  v41 = *(_QWORD *)(v12 + 800);
  if ( !v13 )
    goto LABEL_17;
  v44 = v11;
  v38 = 1;
  v37 = v13 - 1;
  v36 = v12;
  v39 = v8;
  v40 = v10;
  for ( i = (v13 - 1) & sub_AFB9D0(&v44, &v50, v51); ; i = v18 & v37 )
  {
    v15 = (__int64 *)(v41 + 8LL * i);
    v16 = *v15;
    if ( *v15 == -8192 )
      goto LABEL_8;
    if ( v16 == -4096 )
      goto LABEL_30;
    v17 = v50;
    if ( v17 == (unsigned __int16)sub_AF2710(*v15) )
    {
      v25 = sub_AF5140(v16, 0);
      if ( v51[0] == v25 && v49 == *(_DWORD *)(v16 + 4) )
      {
        if ( v46 )
        {
          if ( (*(_BYTE *)(v16 - 16) & 2) != 0 )
            v31 = *(_DWORD *)(v16 - 24);
          else
            v31 = (*(_WORD *)(v16 - 16) >> 6) & 0xF;
          if ( v46 == v31 - 1 )
          {
            v32 = v45;
            v33 = v46;
            v34 = sub_AF15A0((_BYTE *)(v16 - 16)) + 8;
            v35 = &v32[v33];
            while ( *v32 == *v34 )
            {
              ++v32;
              ++v34;
              if ( v35 == v32 )
                goto LABEL_15;
            }
          }
          goto LABEL_7;
        }
        v26 = (*(_BYTE *)(v16 - 16) & 2) != 0 ? *(_DWORD *)(v16 - 24) : (*(_WORD *)(v16 - 16) >> 6) & 0xF;
        if ( v26 - 1 == v48 )
          break;
      }
    }
LABEL_7:
    v16 = *v15;
LABEL_8:
    if ( v16 == -4096 )
    {
LABEL_30:
      v9 = a1;
      v10 = v40;
      v8 = v39;
      goto LABEL_17;
    }
    v18 = i + v38++;
  }
  v27 = v47;
  v28 = v48;
  v29 = sub_AF15A0((_BYTE *)(v16 - 16)) + 8;
  v30 = &v27[v28];
  if ( v30 != v27 )
  {
    while ( *v27 == *v29 )
    {
      ++v27;
      ++v29;
      if ( v30 == v27 )
        goto LABEL_15;
    }
    goto LABEL_7;
  }
LABEL_15:
  v24 = (__int64 *)(v41 + 8LL * i);
  v10 = v40;
  v9 = a1;
  v8 = v39;
  if ( v24 == (__int64 *)(*(_QWORD *)(v36 + 800) + 8LL * *(unsigned int *)(v36 + 816)) || (result = *v24) == 0 )
  {
LABEL_17:
    result = 0;
    if ( a7 )
    {
      v19 = v49;
LABEL_11:
      v20 = *v9;
      v45 = (_QWORD *)a3;
      v21 = v20 + 792;
      v22 = sub_B97910(16, v10 + 1, v8);
      if ( v22 )
      {
        sub_B971C0(v22, (_DWORD)v9, 9, v8, (unsigned int)&v45, 1, a4, v10);
        *(_WORD *)(v22 + 2) = v43;
        *(_DWORD *)(v22 + 4) = v19;
      }
      return sub_B028C0(v22, v8, v21);
    }
  }
  return result;
}
