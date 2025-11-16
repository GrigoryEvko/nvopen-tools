// Function: sub_D8E2A0
// Address: 0xd8e2a0
//
__int64 __fastcall sub_D8E2A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v4; // ecx
  __int64 v5; // r12
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned int v10; // eax
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 *v15; // r10
  int v16; // r11d
  unsigned int v17; // eax
  __int64 *v18; // rcx
  __int64 v19; // rdi
  unsigned int v20; // esi
  int v21; // edx
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned int v24; // esi
  int v25; // eax
  __int64 *v26; // rdx
  int v27; // eax
  int v28; // eax
  __int64 v30; // [rsp+10h] [rbp-A0h]
  char v31; // [rsp+23h] [rbp-8Dh]
  int v32; // [rsp+24h] [rbp-8Ch]
  int v33; // [rsp+28h] [rbp-88h]
  char v34; // [rsp+2Fh] [rbp-81h]
  __int64 v35; // [rsp+38h] [rbp-78h] BYREF
  __int64 v36; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v37; // [rsp+48h] [rbp-68h]
  __int64 v38; // [rsp+50h] [rbp-60h]
  unsigned int v39; // [rsp+58h] [rbp-58h]
  __int64 *v40; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v41; // [rsp+68h] [rbp-48h]
  __int64 v42; // [rsp+70h] [rbp-40h]
  int v43; // [rsp+78h] [rbp-38h]

  result = a3 + 56;
  v4 = *(_DWORD *)(a3 + 96);
  v5 = *(_QWORD *)(a3 + 72);
  v35 = a2;
  v33 = v4;
  v30 = a3 + 56;
  v32 = qword_4F88188;
  v31 = 0;
  if ( v5 != a3 + 56 )
  {
    do
    {
      v7 = *(_QWORD *)(v5 + 144);
      if ( v7 != v5 + 128 )
      {
        v34 = 0;
        do
        {
          sub_D88EE0((__int64)&v36, a1, *(_QWORD *)(v7 + 32), *(_DWORD *)(v7 + 40), v7 + 48);
          if ( !(unsigned __int8)sub_AB1BB0(v5 + 40, (__int64)&v36) )
          {
            if ( v33 > v32 )
            {
              if ( *(_DWORD *)(v5 + 48) <= 0x40u && *(_DWORD *)(a1 + 56) <= 0x40u )
              {
                *(_QWORD *)(v5 + 40) = *(_QWORD *)(a1 + 48);
                *(_DWORD *)(v5 + 48) = *(_DWORD *)(a1 + 56);
              }
              else
              {
                sub_C43990(v5 + 40, a1 + 48);
              }
              if ( *(_DWORD *)(v5 + 64) <= 0x40u && *(_DWORD *)(a1 + 72) <= 0x40u )
              {
                v34 = 1;
                *(_QWORD *)(v5 + 56) = *(_QWORD *)(a1 + 64);
                *(_DWORD *)(v5 + 64) = *(_DWORD *)(a1 + 72);
              }
              else
              {
                sub_C43990(v5 + 56, a1 + 64);
                v34 = 1;
              }
            }
            else
            {
              sub_D87290((__int64)&v40, v5 + 40, (__int64)&v36);
              if ( *(_DWORD *)(v5 + 48) > 0x40u )
              {
                v8 = *(_QWORD *)(v5 + 40);
                if ( v8 )
                  j_j___libc_free_0_0(v8);
              }
              *(_QWORD *)(v5 + 40) = v40;
              *(_DWORD *)(v5 + 48) = v41;
              v41 = 0;
              if ( *(_DWORD *)(v5 + 64) > 0x40u && (v9 = *(_QWORD *)(v5 + 56)) != 0 )
              {
                j_j___libc_free_0_0(v9);
                v10 = v41;
                *(_QWORD *)(v5 + 56) = v42;
                *(_DWORD *)(v5 + 64) = v43;
                if ( v10 > 0x40 && v40 )
                  j_j___libc_free_0_0(v40);
              }
              else
              {
                *(_QWORD *)(v5 + 56) = v42;
                *(_DWORD *)(v5 + 64) = v43;
              }
              v34 = 1;
            }
          }
          if ( v39 > 0x40 && v38 )
            j_j___libc_free_0_0(v38);
          if ( v37 > 0x40 && v36 )
            j_j___libc_free_0_0(v36);
          v7 = sub_220EEE0(v7);
        }
        while ( v5 + 128 != v7 );
        v31 |= v34;
      }
      result = sub_220EEE0(v5);
      v5 = result;
    }
    while ( v30 != result );
    if ( v31 )
    {
      if ( (unsigned __int8)sub_D8CCB0(a1 + 80, &v35, &v36) )
      {
        v11 = *(__int64 **)(v36 + 8);
        v12 = &v11[*(unsigned int *)(v36 + 16)];
        if ( v12 == v11 )
        {
LABEL_49:
          ++*(_DWORD *)(a3 + 96);
          return a3;
        }
        while ( 1 )
        {
          v20 = *(_DWORD *)(a1 + 136);
          if ( !v20 )
            break;
          v13 = v20 - 1;
          v14 = *(_QWORD *)(a1 + 120);
          v15 = 0;
          v16 = 1;
          v17 = v13 & (((unsigned int)*v11 >> 9) ^ ((unsigned int)*v11 >> 4));
          v18 = (__int64 *)(v14 + 8LL * v17);
          v19 = *v18;
          if ( *v18 == *v11 )
          {
LABEL_39:
            if ( v12 == ++v11 )
              goto LABEL_49;
          }
          else
          {
            while ( v19 != -4096 )
            {
              if ( v15 || v19 != -8192 )
                v18 = v15;
              v17 = v13 & (v16 + v17);
              v19 = *(_QWORD *)(v14 + 8LL * v17);
              if ( *v11 == v19 )
                goto LABEL_39;
              ++v16;
              v15 = v18;
              v18 = (__int64 *)(v14 + 8LL * v17);
            }
            v28 = *(_DWORD *)(a1 + 128);
            if ( !v15 )
              v15 = v18;
            ++*(_QWORD *)(a1 + 112);
            v21 = v28 + 1;
            v40 = v15;
            if ( 4 * (v28 + 1) < 3 * v20 )
            {
              if ( v20 - *(_DWORD *)(a1 + 132) - v21 > v20 >> 3 )
                goto LABEL_44;
              goto LABEL_43;
            }
LABEL_42:
            v20 *= 2;
LABEL_43:
            sub_D8E0D0(a1 + 112, v20);
            sub_D8D1D0(a1 + 112, v11, &v40);
            v15 = v40;
            v21 = *(_DWORD *)(a1 + 128) + 1;
LABEL_44:
            *(_DWORD *)(a1 + 128) = v21;
            if ( *v15 != -4096 )
              --*(_DWORD *)(a1 + 132);
            v22 = *v11;
            *v15 = *v11;
            v23 = *(unsigned int *)(a1 + 152);
            if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 156) )
            {
              sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), v23 + 1, 8u, v14, v13);
              v23 = *(unsigned int *)(a1 + 152);
            }
            ++v11;
            *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v23) = v22;
            ++*(_DWORD *)(a1 + 152);
            if ( v12 == v11 )
              goto LABEL_49;
          }
        }
        ++*(_QWORD *)(a1 + 112);
        v40 = 0;
        goto LABEL_42;
      }
      v24 = *(_DWORD *)(a1 + 104);
      v25 = *(_DWORD *)(a1 + 96);
      v26 = (__int64 *)v36;
      ++*(_QWORD *)(a1 + 80);
      v27 = v25 + 1;
      v40 = v26;
      if ( 4 * v27 >= 3 * v24 )
      {
        v24 *= 2;
      }
      else if ( v24 - *(_DWORD *)(a1 + 100) - v27 > v24 >> 3 )
      {
LABEL_52:
        *(_DWORD *)(a1 + 96) = v27;
        if ( *v26 != -4096 )
          --*(_DWORD *)(a1 + 100);
        *v26 = v35;
        v26[1] = (__int64)(v26 + 3);
        v26[2] = 0x400000000LL;
        ++*(_DWORD *)(a3 + 96);
        return a3;
      }
      sub_D8CE70(a1 + 80, v24);
      sub_D8CCB0(a1 + 80, &v35, &v40);
      v26 = v40;
      v27 = *(_DWORD *)(a1 + 96) + 1;
      goto LABEL_52;
    }
  }
  return result;
}
