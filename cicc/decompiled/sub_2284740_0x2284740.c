// Function: sub_2284740
// Address: 0x2284740
//
__int64 __fastcall sub_2284740(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  __int64 *v4; // r12
  __int64 v5; // r13
  char v6; // bl
  __int64 v7; // rdx
  char v8; // si
  __int64 v9; // rdi
  int v10; // ecx
  unsigned int v11; // eax
  __int64 *v12; // rbx
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 i; // r15
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v22; // ecx
  unsigned int v23; // eax
  int v24; // edi
  __int64 v25; // rbx
  int v26; // r8d
  __int64 *v27; // r10
  unsigned int v28; // esi
  __int64 v29; // [rsp+0h] [rbp-A0h]
  __int64 *v30; // [rsp+10h] [rbp-90h]
  __int64 *v33; // [rsp+28h] [rbp-78h]
  _QWORD v34[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v35; // [rsp+40h] [rbp-60h]
  __int64 v36; // [rsp+50h] [rbp-50h] BYREF
  __int64 v37; // [rsp+58h] [rbp-48h] BYREF
  __int64 v38; // [rsp+60h] [rbp-40h]
  __int64 v39; // [rsp+68h] [rbp-38h]

  v3 = (_QWORD *)(a1 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 1;
  v29 = a1 + 16;
  do
  {
    if ( v3 )
      *v3 = -4096;
    v3 += 2;
  }
  while ( v3 != (_QWORD *)(a1 + 80) );
  v4 = *(__int64 **)(a2 + 8);
  v33 = v4;
  v30 = &v4[*(unsigned int *)(a2 + 16)];
  if ( v4 != v30 )
  {
    do
    {
      v5 = *v33;
      v6 = *(_BYTE *)(a1 + 8);
      v7 = *(_QWORD *)(*v33 + 8);
      v37 = 0;
      v36 = v7;
      v8 = v6 & 1;
      if ( (v6 & 1) != 0 )
      {
        v9 = v29;
        v10 = 3;
      }
      else
      {
        v22 = *(_DWORD *)(a1 + 24);
        v9 = *(_QWORD *)(a1 + 16);
        if ( !v22 )
        {
          v34[0] = 0;
          ++*(_QWORD *)a1;
          v23 = *(_DWORD *)(a1 + 8);
          v24 = (v23 >> 1) + 1;
          goto LABEL_49;
        }
        v10 = v22 - 1;
      }
      v11 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v7 == *v12 )
        goto LABEL_9;
      v26 = 1;
      v27 = 0;
      while ( v13 != -4096 )
      {
        if ( v13 == -8192 && !v27 )
          v27 = v12;
        v11 = v10 & (v26 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v7 == *v12 )
          goto LABEL_9;
        ++v26;
      }
      if ( !v27 )
        v27 = v12;
      ++*(_QWORD *)a1;
      v23 = *(_DWORD *)(a1 + 8);
      v34[0] = v27;
      v24 = (v23 >> 1) + 1;
      if ( !v8 )
      {
        v22 = *(_DWORD *)(a1 + 24);
LABEL_49:
        if ( 4 * v24 >= 3 * v22 )
          goto LABEL_60;
        goto LABEL_50;
      }
      v22 = 4;
      if ( (unsigned int)(4 * v24) >= 0xC )
      {
LABEL_60:
        v25 = a1;
        v28 = 2 * v22;
LABEL_61:
        sub_227A910(v25, v28);
        sub_227A6D0(v25, &v36, v34);
        v7 = v36;
        v23 = *(_DWORD *)(v25 + 8);
        goto LABEL_51;
      }
LABEL_50:
      v25 = a1;
      if ( v22 - *(_DWORD *)(a1 + 12) - v24 <= v22 >> 3 )
      {
        v28 = v22;
        goto LABEL_61;
      }
LABEL_51:
      v12 = (__int64 *)v34[0];
      *(_DWORD *)(a1 + 8) = (2 * (v23 >> 1) + 2) | v23 & 1;
      if ( *v12 != -4096 )
        --*(_DWORD *)(a1 + 12);
      *v12 = v7;
      v12[1] = v37;
LABEL_9:
      v14 = *(_QWORD *)(v5 + 8);
      v15 = *(_QWORD *)(v14 + 80);
      v16 = v14 + 72;
      if ( v14 + 72 == v15 )
      {
        i = 0;
      }
      else
      {
        if ( !v15 )
          BUG();
        while ( 1 )
        {
          i = *(_QWORD *)(v15 + 32);
          if ( i != v15 + 24 )
            break;
          v15 = *(_QWORD *)(v15 + 8);
          if ( v16 == v15 )
            break;
          if ( !v15 )
            BUG();
        }
      }
LABEL_15:
      while ( v16 != v15 )
      {
        if ( !i )
          BUG();
        if ( (unsigned __int8)(*(_BYTE *)(i - 24) - 34) <= 0x33u )
        {
          v18 = 0x8000000000041LL;
          if ( _bittest64(&v18, (unsigned int)*(unsigned __int8 *)(i - 24) - 34) )
          {
            v19 = *(_QWORD *)(i - 56);
            if ( v19 && !*(_BYTE *)v19 && *(_QWORD *)(v19 + 24) == *(_QWORD *)(i + 56) )
            {
              ++*((_DWORD *)v12 + 2);
            }
            else
            {
              ++*((_DWORD *)v12 + 3);
              v34[0] = 6;
              v34[1] = 0;
              v35 = i - 24;
              if ( i == -8168 || i == -4072 )
              {
                v36 = i - 24;
                v39 = i - 24;
                v37 = 6;
                v38 = 0;
              }
              else
              {
                sub_BD73F0((__int64)v34);
                v36 = i - 24;
                v37 = 6;
                v38 = 0;
                v39 = v35;
                if ( v35 != -8192 && v35 != -4096 && v35 )
                  sub_BD6050((unsigned __int64 *)&v37, v34[0] & 0xFFFFFFFFFFFFFFF8LL);
              }
              sub_2282310(a3, &v36, &v37);
              if ( v39 != 0 && v39 != -4096 && v39 != -8192 )
                sub_BD60C0(&v37);
              if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
                sub_BD60C0(v34);
            }
          }
        }
        for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v15 + 32) )
        {
          v20 = v15 - 24;
          if ( !v15 )
            v20 = 0;
          if ( i != v20 + 48 )
            break;
          v15 = *(_QWORD *)(v15 + 8);
          if ( v16 == v15 )
            goto LABEL_15;
          if ( !v15 )
            BUG();
        }
      }
      ++v33;
    }
    while ( v30 != v33 );
  }
  return a1;
}
