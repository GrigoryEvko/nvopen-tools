// Function: sub_2AD3790
// Address: 0x2ad3790
//
void __fastcall sub_2AD3790(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 *v5; // r15
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 *v9; // r13
  __int64 v10; // r9
  int v11; // r11d
  __int64 *v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 *v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // esi
  int v19; // ecx
  int v20; // eax
  __int64 v21; // [rsp-C0h] [rbp-C0h]
  __int64 v22; // [rsp-A0h] [rbp-A0h]
  __int64 v23; // [rsp-90h] [rbp-90h] BYREF
  _QWORD v24[4]; // [rsp-88h] [rbp-88h] BYREF
  __int64 *v25; // [rsp-68h] [rbp-68h] BYREF
  int v26; // [rsp-60h] [rbp-60h]
  __int64 v27; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_DWORD *)(a1 + 280) == *(_DWORD *)(a1 + 276) )
  {
    v2 = *(_QWORD *)(a1 + 440);
    v3 = *(_QWORD *)(v2 + 112);
    v22 = v3 + 184LL * *(unsigned int *)(v2 + 120);
    if ( v3 != v22 )
    {
      while ( 1 )
      {
        v4 = *(_QWORD *)v3;
        if ( *(_QWORD *)(v3 + 64) != *(_QWORD *)(*(_QWORD *)v3 + 8LL) )
          goto LABEL_4;
        sub_1022EF0(*(_DWORD *)(v3 + 48));
        if ( !byte_500D668
          && ((unsigned __int8)sub_31A4BE0(*(_QWORD *)(a1 + 496)) || !*(_BYTE *)(v3 + 73))
          && !(unsigned __int8)sub_DFE340(*(_QWORD *)(a1 + 448)) )
        {
          goto LABEL_4;
        }
        sub_1022F10(&v25, v3 + 8, v4);
        if ( v26 )
          break;
LABEL_10:
        v5 = v25;
LABEL_11:
        if ( v5 == &v27 )
        {
LABEL_4:
          v3 += 184;
          if ( v22 == v3 )
            return;
        }
        else
        {
          v3 += 184;
          _libc_free((unsigned __int64)v5);
          if ( v22 == v3 )
            return;
        }
      }
      v6 = sub_AE6EC0(a1 + 256, v4);
      if ( *(_BYTE *)(a1 + 284) )
        v7 = *(unsigned int *)(a1 + 276);
      else
        v7 = *(unsigned int *)(a1 + 272);
      v8 = *(_QWORD *)(a1 + 264) + 8 * v7;
      v24[0] = v6;
      v24[1] = v8;
      sub_254BBF0((__int64)v24);
      v9 = v25;
      v5 = &v25[v26];
      if ( v25 == v5 )
        goto LABEL_11;
      v21 = a1 + 320;
      while ( 1 )
      {
        v17 = *v9;
        v18 = *(_DWORD *)(a1 + 344);
        v23 = *v9;
        if ( !v18 )
          break;
        v10 = *(_QWORD *)(a1 + 328);
        v11 = 1;
        v12 = 0;
        v13 = (v18 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v14 = (__int64 *)(v10 + 16LL * v13);
        v15 = *v14;
        if ( v17 != *v14 )
        {
          while ( v15 != -4096 )
          {
            if ( v15 == -8192 && !v12 )
              v12 = v14;
            v13 = (v18 - 1) & (v11 + v13);
            v14 = (__int64 *)(v10 + 16LL * v13);
            v15 = *v14;
            if ( v17 == *v14 )
              goto LABEL_22;
            ++v11;
          }
          if ( !v12 )
            v12 = v14;
          v20 = *(_DWORD *)(a1 + 336);
          ++*(_QWORD *)(a1 + 320);
          v19 = v20 + 1;
          v24[0] = v12;
          if ( 4 * (v20 + 1) < 3 * v18 )
          {
            if ( v18 - *(_DWORD *)(a1 + 340) - v19 <= v18 >> 3 )
            {
LABEL_27:
              sub_2978470(v21, v18);
              sub_2AC1910(v21, &v23, v24);
              v17 = v23;
              v12 = (__int64 *)v24[0];
              v19 = *(_DWORD *)(a1 + 336) + 1;
            }
            *(_DWORD *)(a1 + 336) = v19;
            if ( *v12 != -4096 )
              --*(_DWORD *)(a1 + 340);
            *v12 = v17;
            v16 = v12 + 1;
            v12[1] = 0;
            goto LABEL_23;
          }
LABEL_26:
          v18 *= 2;
          goto LABEL_27;
        }
LABEL_22:
        v16 = v14 + 1;
LABEL_23:
        ++v9;
        *v16 = v4;
        v4 = v23;
        if ( v5 == v9 )
          goto LABEL_10;
      }
      ++*(_QWORD *)(a1 + 320);
      v24[0] = 0;
      goto LABEL_26;
    }
  }
}
