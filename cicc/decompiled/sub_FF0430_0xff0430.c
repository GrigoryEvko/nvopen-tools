// Function: sub_FF0430
// Address: 0xff0430
//
__int64 __fastcall sub_FF0430(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // edx
  int v5; // edi
  unsigned int i; // eax
  __int64 *v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // eax
  unsigned __int64 v10; // rax
  __int64 v11; // r14
  int v12; // r13d
  unsigned int v13; // r15d
  unsigned int v14; // ebx
  __int64 v15; // rax
  unsigned int v16; // esi
  unsigned int v18; // ebx
  unsigned __int64 v19; // rax
  __int64 v20; // r14
  int v21; // r13d
  unsigned int v22; // r15d
  int v23; // edi
  unsigned int j; // eax
  __int64 v25; // rcx
  unsigned int v26; // eax
  __int64 v27; // rsi
  unsigned __int64 v28; // rcx
  __int64 v29; // [rsp+8h] [rbp-68h]
  unsigned int v30; // [rsp+10h] [rbp-60h]
  unsigned __int64 v31; // [rsp+10h] [rbp-60h]
  unsigned int v32; // [rsp+1Ch] [rbp-54h]
  __int64 v33; // [rsp+20h] [rbp-50h]
  __int64 v34; // [rsp+28h] [rbp-48h]
  unsigned int v35; // [rsp+28h] [rbp-48h]
  _DWORD v36[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v4 = *(_DWORD *)(a1 + 56);
  v34 = *(_QWORD *)(a1 + 40);
  if ( !v4 )
  {
LABEL_7:
    v10 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10 == a2 + 48 )
      goto LABEL_33;
    if ( !v10 )
      goto LABEL_34;
    v11 = v10 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
    {
LABEL_33:
      sub_F02DB0(v36, 0, 0);
    }
    else
    {
      v35 = sub_B46E30(v11);
      v12 = sub_B46E30(v11);
      if ( v12 )
      {
        v13 = 0;
        v14 = 0;
        do
        {
          v15 = sub_B46EC0(v11, v13++);
          v14 += a3 == v15;
        }
        while ( v12 != v13 );
        v16 = v14;
      }
      else
      {
        v16 = 0;
      }
      sub_F02DB0(v36, v16, v35);
    }
    return v36[0];
  }
  v5 = 1;
  v32 = v4 - 1;
  for ( i = (v4 - 1) & (969526130 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))); ; i = v32 & v9 )
  {
    v7 = (__int64 *)(v34 + 24LL * i);
    v8 = *v7;
    if ( a2 == *v7 )
    {
      v18 = *((_DWORD *)v7 + 2);
      if ( !v18 )
        break;
    }
    if ( v8 == -4096 && *((_DWORD *)v7 + 2) == -1 )
      goto LABEL_7;
    v9 = v5 + i;
    ++v5;
  }
  v33 = *v7;
  v19 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v19 == v8 + 48 )
    return v18;
  if ( !v19 )
LABEL_34:
    BUG();
  v20 = v19 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 <= 0xA )
  {
    v30 = v4;
    v21 = sub_B46E30(v20);
    if ( v21 )
    {
      v22 = 0;
      v29 = v34 + 24LL * v30;
      do
      {
        if ( a3 == sub_B46EC0(v20, v22) )
        {
          v23 = 1;
          v31 = (unsigned __int64)(((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4)) << 32;
          for ( j = v32 & (((0xBF58476D1CE4E5B9LL * (v31 | (37 * v22))) >> 31) ^ (484763065 * (v31 | (37 * v22))));
                ;
                j = v32 & v26 )
          {
            v25 = v34 + 24LL * j;
            if ( v33 == *(_QWORD *)v25 && *(_DWORD *)(v25 + 8) == v22 )
              break;
            if ( *(_QWORD *)v25 == -4096 && *(_DWORD *)(v25 + 8) == -1 )
            {
              v25 = v29;
              break;
            }
            v26 = v23 + j;
            ++v23;
          }
          v27 = *(unsigned int *)(v25 + 16);
          v28 = v27 + v18;
          v18 += v27;
          if ( v28 > 0x80000000 )
            v18 = 0x80000000;
        }
        ++v22;
      }
      while ( v21 != v22 );
    }
  }
  return v18;
}
