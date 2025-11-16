// Function: sub_9BE8F0
// Address: 0x9be8f0
//
__int64 __fastcall sub_9BE8F0(unsigned int *a1, __int64 a2, signed int a3, unsigned __int8 a4)
{
  unsigned __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // rdx
  unsigned int v10; // eax
  int v11; // ecx
  signed int v13; // r8d
  int v14; // r11d
  _DWORD *v15; // rcx
  unsigned int v16; // r8d
  _DWORD *v17; // rax
  int v18; // edi
  _QWORD *v19; // rax
  int v20; // r9d
  unsigned int v21; // eax
  int v22; // eax
  unsigned int v23; // eax
  unsigned int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // edx
  int v27; // edi
  int v28; // r10d
  _DWORD *v29; // r9
  unsigned int v30; // eax
  unsigned int v31; // edx
  __int64 v32; // rdi
  int v33; // r9d
  unsigned int v34; // r14d
  _DWORD *v35; // r8
  int v36; // esi
  unsigned __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+18h] [rbp-38h]

  v38 = sub_9B6BB0(a3, a1[10], (void (__fastcall *)(int **, __int64 *, __int64 *, char *))sub_C45F70);
  v7 = HIDWORD(v38);
  if ( BYTE4(v38) )
  {
    if ( (unsigned int)(v38 + 0x7FFFFFFF) > 0xFFFFFFFD )
    {
LABEL_5:
      LODWORD(v7) = 0;
      return (unsigned int)v7;
    }
    v8 = a1[8];
    v9 = *((_QWORD *)a1 + 2);
    if ( v8 )
    {
      v10 = (v8 - 1) & (37 * v38);
      v11 = *(_DWORD *)(v9 + 16LL * v10);
      if ( v11 == (_DWORD)v38 )
        goto LABEL_5;
      v20 = 1;
      while ( v11 != 0x7FFFFFFF )
      {
        v10 = (v8 - 1) & (v20 + v10);
        v11 = *(_DWORD *)(v9 + 16LL * v10);
        if ( v11 == (_DWORD)v38 )
          goto LABEL_5;
        ++v20;
      }
    }
    v13 = a1[11];
    if ( v13 >= (int)v38 )
    {
      if ( (int)a1[10] > (int)v38 )
      {
        v39 = sub_9B6BB0(v13, v38, (void (__fastcall *)(int **, __int64 *, __int64 *, char *))sub_C46BD0);
        if ( !BYTE4(v39) || (int)v39 >= (__int64)*a1 )
          goto LABEL_5;
        v9 = *((_QWORD *)a1 + 2);
        v8 = a1[8];
        a1[10] = v38;
      }
    }
    else
    {
      if ( (int)*a1 <= a3 )
        goto LABEL_5;
      a1[11] = v38;
    }
    if ( *((_BYTE *)a1 + 5) > a4 )
      *((_BYTE *)a1 + 5) = a4;
    if ( v8 )
    {
      v14 = 1;
      v15 = 0;
      v16 = (v8 - 1) & (37 * v38);
      v17 = (_DWORD *)(v9 + 16LL * v16);
      v18 = *v17;
      if ( *v17 == (_DWORD)v38 )
      {
LABEL_14:
        v19 = v17 + 2;
LABEL_15:
        *v19 = a2;
        return (unsigned int)v7;
      }
      while ( v18 != 0x7FFFFFFF )
      {
        if ( v18 == 0x80000000 && !v15 )
          v15 = v17;
        v16 = (v8 - 1) & (v14 + v16);
        v17 = (_DWORD *)(v9 + 16LL * v16);
        v18 = *v17;
        if ( *v17 == (_DWORD)v38 )
          goto LABEL_14;
        ++v14;
      }
      if ( !v15 )
        v15 = v17;
      v21 = a1[6];
      ++*((_QWORD *)a1 + 1);
      v22 = v21 + 1;
      if ( 4 * v22 < 3 * v8 )
      {
        if ( v8 - (v22 + a1[7]) > v8 >> 3 )
        {
LABEL_34:
          a1[6] = v22;
          if ( *v15 != 0x7FFFFFFF )
            --a1[7];
          *v15 = v38;
          v19 = v15 + 2;
          *((_QWORD *)v15 + 1) = 0;
          goto LABEL_15;
        }
        sub_9BE710((__int64)(a1 + 2), v8);
        v30 = a1[8];
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *((_QWORD *)a1 + 2);
          v33 = 1;
          v34 = (v30 - 1) & (37 * v38);
          v35 = 0;
          v22 = a1[6] + 1;
          v15 = (_DWORD *)(v32 + 16LL * v34);
          v36 = *v15;
          if ( (_DWORD)v38 != *v15 )
          {
            while ( v36 != 0x7FFFFFFF )
            {
              if ( v36 == 0x80000000 && !v35 )
                v35 = v15;
              v34 = v31 & (v33 + v34);
              v15 = (_DWORD *)(v32 + 16LL * v34);
              v36 = *v15;
              if ( *v15 == (_DWORD)v38 )
                goto LABEL_34;
              ++v33;
            }
            if ( v35 )
              v15 = v35;
          }
          goto LABEL_34;
        }
LABEL_61:
        ++a1[6];
        BUG();
      }
    }
    else
    {
      ++*((_QWORD *)a1 + 1);
    }
    sub_9BE710((__int64)(a1 + 2), 2 * v8);
    v23 = a1[8];
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *((_QWORD *)a1 + 2);
      v26 = (v23 - 1) & (37 * v38);
      v22 = a1[6] + 1;
      v15 = (_DWORD *)(v25 + 16LL * v26);
      v27 = *v15;
      if ( *v15 != (_DWORD)v38 )
      {
        v28 = 1;
        v29 = 0;
        while ( v27 != 0x7FFFFFFF )
        {
          if ( !v29 && v27 == 0x80000000 )
            v29 = v15;
          v26 = v24 & (v28 + v26);
          v15 = (_DWORD *)(v25 + 16LL * v26);
          v27 = *v15;
          if ( *v15 == (_DWORD)v38 )
            goto LABEL_34;
          ++v28;
        }
        if ( v29 )
          v15 = v29;
      }
      goto LABEL_34;
    }
    goto LABEL_61;
  }
  return (unsigned int)v7;
}
