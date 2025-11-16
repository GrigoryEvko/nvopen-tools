// Function: sub_1AA6640
// Address: 0x1aa6640
//
__int64 __fastcall sub_1AA6640(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 *v9; // rdx
  __int64 v10; // r12
  __int64 *v11; // r14
  __int64 v12; // rdi
  char v13; // al
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  char v20; // r10
  unsigned int v21; // r9d
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  _QWORD *v26; // r13
  __int64 v27; // rdi
  unsigned __int64 v28; // rsi
  __int64 v29; // rsi
  __int64 v31; // rax
  char v32; // r10
  unsigned int v33; // r9d
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rdi
  unsigned __int64 v40; // rsi
  __int64 v41; // rsi
  __int64 v42; // rsi
  unsigned __int64 v43; // rdi
  _QWORD *v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  __int64 v47; // [rsp+20h] [rbp-40h]

  v3 = a3;
  v4 = a3 + 40;
  v44 = (_QWORD *)sub_157EBA0(a3);
  v5 = sub_15F4880(a1);
  sub_157E9D0(v4, v5);
  v6 = *(_QWORD *)(v3 + 40);
  v7 = *(_QWORD *)(v5 + 24);
  *(_QWORD *)(v5 + 32) = v4;
  v6 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v5 + 24) = v6 | v7 & 7;
  *(_QWORD *)(v6 + 8) = v5 + 24;
  *(_QWORD *)(v3 + 40) = (v5 + 24) | *(_QWORD *)(v3 + 40) & 7LL;
  if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
  {
    v8 = *(__int64 **)(v5 - 8);
    v9 = &v8[3 * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)];
  }
  else
  {
    v9 = (__int64 *)v5;
    v8 = (__int64 *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
  }
  if ( v8 != v9 )
  {
    v45 = v3;
    v10 = v5;
    v11 = v9;
    v46 = v5 + 24;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v8;
        v13 = *(_BYTE *)(*v8 + 16);
        if ( v13 != 71 )
        {
          if ( v13 == 77 && a2 == *(_QWORD *)(v12 + 40) )
          {
            v31 = 0x17FFFFFFE8LL;
            v32 = *(_BYTE *)(v12 + 23) & 0x40;
            v33 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
            if ( v33 )
            {
              v34 = 24LL * *(unsigned int *)(v12 + 56) + 8;
              v35 = 0;
              do
              {
                v36 = v12 - 24LL * v33;
                if ( v32 )
                  v36 = *(_QWORD *)(v12 - 8);
                if ( v45 == *(_QWORD *)(v36 + v34) )
                {
                  v31 = 24 * v35;
                  goto LABEL_42;
                }
                ++v35;
                v34 += 8;
              }
              while ( v33 != (_DWORD)v35 );
              v31 = 0x17FFFFFFE8LL;
            }
LABEL_42:
            if ( v32 )
              v37 = *(_QWORD *)(v12 - 8);
            else
              v37 = v12 - 24LL * v33;
            v38 = *(_QWORD *)(v37 + v31);
            if ( v38 )
            {
              v39 = v8[1];
              v40 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v40 = v39;
              if ( v39 )
                *(_QWORD *)(v39 + 16) = *(_QWORD *)(v39 + 16) & 3LL | v40;
              *v8 = v38;
              v41 = *(_QWORD *)(v38 + 8);
              v8[1] = v41;
              if ( v41 )
                *(_QWORD *)(v41 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v41 + 16) & 3LL;
              v8[2] = (v38 + 8) | v8[2] & 3;
              *(_QWORD *)(v38 + 8) = v8;
            }
            else
            {
              v42 = v8[1];
              v43 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v43 = v42;
              if ( v42 )
                *(_QWORD *)(v42 + 16) = v43 | *(_QWORD *)(v42 + 16) & 3LL;
              *v8 = 0;
            }
          }
          goto LABEL_6;
        }
        v47 = *(_QWORD *)(v12 - 24);
        v14 = sub_15F4880(v12);
        sub_157E9D0(v4, v14);
        v15 = *(_QWORD *)(v10 + 24);
        *(_QWORD *)(v14 + 32) = v46;
        v15 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v14 + 24) = v15 | *(_QWORD *)(v14 + 24) & 7LL;
        *(_QWORD *)(v15 + 8) = v14 + 24;
        *(_QWORD *)(v10 + 24) = *(_QWORD *)(v10 + 24) & 7LL | (v14 + 24);
        if ( *v8 )
        {
          v16 = v8[1];
          v17 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v17 = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
        }
        *v8 = v14;
        v18 = *(_QWORD *)(v14 + 8);
        v8[1] = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v18 + 16) & 3LL;
        v8[2] = (v14 + 8) | v8[2] & 3;
        *(_QWORD *)(v14 + 8) = v8;
        if ( *(_BYTE *)(v47 + 16) == 77 && a2 == *(_QWORD *)(v47 + 40) )
          break;
LABEL_6:
        v8 += 3;
        if ( v8 == v11 )
          goto LABEL_32;
      }
      v19 = 0x17FFFFFFE8LL;
      v20 = *(_BYTE *)(v47 + 23) & 0x40;
      v21 = *(_DWORD *)(v47 + 20) & 0xFFFFFFF;
      if ( v21 )
      {
        v22 = 24LL * *(unsigned int *)(v47 + 56) + 8;
        v23 = 0;
        do
        {
          v24 = v47 - 24LL * v21;
          if ( v20 )
            v24 = *(_QWORD *)(v47 - 8);
          if ( v45 == *(_QWORD *)(v24 + v22) )
          {
            v19 = 24 * v23;
            goto LABEL_22;
          }
          ++v23;
          v22 += 8;
        }
        while ( v21 != (_DWORD)v23 );
        v19 = 0x17FFFFFFE8LL;
        if ( !v20 )
        {
LABEL_23:
          v25 = *(_QWORD *)(v47 - 24LL * v21 + v19);
          if ( (*(_BYTE *)(v14 + 23) & 0x40) == 0 )
            goto LABEL_52;
          goto LABEL_24;
        }
      }
      else
      {
LABEL_22:
        if ( !v20 )
          goto LABEL_23;
      }
      v25 = *(_QWORD *)(*(_QWORD *)(v47 - 8) + v19);
      if ( (*(_BYTE *)(v14 + 23) & 0x40) == 0 )
      {
LABEL_52:
        v26 = (_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
        goto LABEL_25;
      }
LABEL_24:
      v26 = *(_QWORD **)(v14 - 8);
LABEL_25:
      if ( *v26 )
      {
        v27 = v26[1];
        v28 = v26[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v28 = v27;
        if ( v27 )
          *(_QWORD *)(v27 + 16) = *(_QWORD *)(v27 + 16) & 3LL | v28;
      }
      *v26 = v25;
      if ( !v25 )
        goto LABEL_6;
      v29 = *(_QWORD *)(v25 + 8);
      v26[1] = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = (unsigned __int64)(v26 + 1) | *(_QWORD *)(v29 + 16) & 3LL;
      v8 += 3;
      v26[2] = (v25 + 8) | v26[2] & 3LL;
      *(_QWORD *)(v25 + 8) = v26;
      if ( v8 == v11 )
      {
LABEL_32:
        v3 = v45;
        v5 = v10;
        break;
      }
    }
  }
  sub_157F2D0(a2, v3, 0);
  sub_15F20C0(v44);
  return v5;
}
