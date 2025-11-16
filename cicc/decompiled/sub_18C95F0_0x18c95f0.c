// Function: sub_18C95F0
// Address: 0x18c95f0
//
void __fastcall sub_18C95F0(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // r15
  __int64 v5; // r13
  __int64 v6; // r12
  _QWORD *v7; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // r9
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // r12
  unsigned __int64 v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rcx
  _QWORD *v21; // rcx
  __int64 v22; // r8
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rbx
  __int64 *v27; // r15
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // rdx
  __int64 v33; // r9
  __int64 *v34; // rax
  __int64 v35; // r15
  unsigned __int64 v36; // r8
  _QWORD *v37; // rax
  _QWORD *v38; // r15
  int v39; // edx
  int v40; // esi
  __int64 v41; // rdx
  unsigned __int64 v42; // rcx
  __int64 v43; // [rsp+0h] [rbp-70h]
  _QWORD *v44; // [rsp+8h] [rbp-68h]
  _QWORD *v45; // [rsp+8h] [rbp-68h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  _BYTE v47[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v48; // [rsp+30h] [rbp-40h]

  v3 = a2;
  if ( (a3 == 17 || a3 > 0x17u) && a2 )
  {
    do
    {
      v5 = *(_QWORD *)(v3 + 8);
      v6 = (unsigned int)sub_1648720(v3);
      if ( sub_15CD290(*(_QWORD *)(a1[1] + 168), v3) && sub_15CD0F0(*(_QWORD *)(a1[1] + 168), *a1, v3) )
      {
        *(_BYTE *)(a1[1] + 153) = 1;
        v7 = (_QWORD *)*a1;
        v46 = **(_QWORD **)v3;
        v8 = sub_1648700(v3);
        v9 = v8;
        if ( *((_BYTE *)v8 + 16) == 77 )
        {
          if ( (*((_BYTE *)v8 + 23) & 0x40) != 0 )
            v16 = (_QWORD *)*(v8 - 1);
          else
            v16 = &v8[-3 * (*((_DWORD *)v8 + 5) & 0xFFFFFFF)];
          v17 = v16[3 * *((unsigned int *)v8 + 14) + 1 + v6];
          if ( v46 != *v7 )
          {
            v25 = a1;
            v26 = v17;
            v45 = v9;
            v27 = v25;
            if ( *(_BYTE *)(sub_157ED20(v17) + 16) == 34 )
            {
              while ( 1 )
              {
                v28 = *(_QWORD *)(v27[1] + 168);
                v29 = *(unsigned int *)(v28 + 48);
                if ( !(_DWORD)v29 )
                  goto LABEL_59;
                v30 = *(_QWORD *)(v28 + 32);
                v31 = (v29 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
                v32 = (__int64 *)(v30 + 16LL * v31);
                v33 = *v32;
                if ( v26 != *v32 )
                  break;
LABEL_44:
                if ( v32 == (__int64 *)(v30 + 16 * v29) )
                  goto LABEL_59;
                v26 = **(_QWORD **)(v32[1] + 8);
                if ( *(_BYTE *)(sub_157ED20(v26) + 16) != 34 )
                  goto LABEL_46;
              }
              v39 = 1;
              while ( v33 != -8 )
              {
                v40 = v39 + 1;
                v31 = (v29 - 1) & (v39 + v31);
                v32 = (__int64 *)(v30 + 16LL * v31);
                v33 = *v32;
                if ( v26 == *v32 )
                  goto LABEL_44;
                v39 = v40;
              }
LABEL_59:
              BUG();
            }
LABEL_46:
            v34 = v27;
            v35 = v26;
            v48 = 257;
            a1 = v34;
            v36 = (*(_QWORD *)(v35 + 40) & 0xFFFFFFFFFFFFFFF8LL) - 24;
            if ( (*(_QWORD *)(v35 + 40) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              v36 = 0;
            v43 = v36;
            v37 = sub_1648A60(56, 1u);
            v9 = v45;
            v38 = v37;
            if ( v37 )
            {
              sub_15FD590((__int64)v37, (__int64)v7, v46, (__int64)v47, v43);
              v9 = v45;
            }
            v7 = v38;
          }
          if ( (*((_DWORD *)v9 + 5) & 0xFFFFFFF) != 0 )
          {
            v3 = v5;
            v18 = 0;
            v19 = 8LL * (*((_DWORD *)v9 + 5) & 0xFFFFFFF);
            do
            {
              if ( (*((_BYTE *)v9 + 23) & 0x40) != 0 )
                v20 = (_QWORD *)*(v9 - 1);
              else
                v20 = &v9[-3 * (*((_DWORD *)v9 + 5) & 0xFFFFFFF)];
              if ( v17 == v20[3 * *((unsigned int *)v9 + 14) + 1 + v18 / 8] )
              {
                v21 = &v20[3 * v18 / 8];
                if ( v3 && (_QWORD *)v3 == v21 )
                  v3 = *(_QWORD *)(v3 + 8);
                if ( *v21 )
                {
                  v22 = v21[1];
                  v23 = v21[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v23 = v22;
                  if ( v22 )
                    *(_QWORD *)(v22 + 16) = *(_QWORD *)(v22 + 16) & 3LL | v23;
                }
                *v21 = v7;
                if ( v7 )
                {
                  v24 = v7[1];
                  v21[1] = v24;
                  if ( v24 )
                    *(_QWORD *)(v24 + 16) = (unsigned __int64)(v21 + 1) | *(_QWORD *)(v24 + 16) & 3LL;
                  v21[2] = (unsigned __int64)(v7 + 1) | v21[2] & 3LL;
                  v7[1] = v21;
                }
              }
              v18 += 8LL;
            }
            while ( v18 != v19 );
            continue;
          }
        }
        else
        {
          if ( v46 == *v7 )
            goto LABEL_10;
          v48 = 257;
          v44 = sub_1648700(v3);
          v10 = sub_1648A60(56, 1u);
          v11 = v10;
          if ( v10 )
          {
            v12 = (__int64)v7;
            v7 = v10;
            sub_15FD590((__int64)v10, v12, v46, (__int64)v47, (__int64)v44);
            if ( *(_QWORD *)v3 )
            {
LABEL_10:
              v13 = *(_QWORD *)(v3 + 8);
              v14 = *(_QWORD *)(v3 + 16) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v14 = v13;
              if ( v13 )
                *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
              *(_QWORD *)v3 = v7;
              v11 = v7;
            }
            else
            {
              *(_QWORD *)v3 = v11;
            }
            v15 = v11[1];
            *(_QWORD *)(v3 + 8) = v15;
            if ( v15 )
              *(_QWORD *)(v15 + 16) = (v3 + 8) | *(_QWORD *)(v15 + 16) & 3LL;
            *(_QWORD *)(v3 + 16) = (unsigned __int64)(v11 + 1) | *(_QWORD *)(v3 + 16) & 3LL;
            v11[1] = v3;
            v3 = v5;
            continue;
          }
          if ( *(_QWORD *)v3 )
          {
            v41 = *(_QWORD *)(v3 + 8);
            v42 = *(_QWORD *)(v3 + 16) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v42 = v41;
            if ( v41 )
              *(_QWORD *)(v41 + 16) = v42 | *(_QWORD *)(v41 + 16) & 3LL;
            *(_QWORD *)v3 = 0;
            v3 = v5;
            continue;
          }
        }
      }
      v3 = v5;
    }
    while ( v3 );
  }
}
