// Function: sub_2365C20
// Address: 0x2365c20
//
__int64 __fastcall sub_2365C20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  unsigned int v11; // r12d
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r12
  __int64 v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rcx
  unsigned __int64 i; // r15
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // rdx
  unsigned int v29; // ebx
  __int64 result; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // esi
  __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // rbx
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  int v42; // eax
  __int64 v43; // rdx
  size_t v44; // rdx
  int v45; // edx
  _QWORD *v46; // [rsp+10h] [rbp-60h]
  _QWORD *v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+20h] [rbp-50h] BYREF
  __int64 v49; // [rsp+28h] [rbp-48h]
  __int64 v50; // [rsp+30h] [rbp-40h]

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_DWORD *)(a2 + 24);
  ++*(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = v8;
  v10 = *(_QWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 24) = v9;
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 16) = v10;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  v11 = *(_DWORD *)(a2 + 40);
  if ( v11 )
  {
    v31 = a1 + 32;
    if ( v31 != a2 + 32 )
    {
      v32 = *(_QWORD *)(a2 + 32);
      v33 = a2 + 48;
      if ( v32 == a2 + 48 )
      {
        sub_2358FB0(v31, v11, v33, a4, a5, a6);
        v34 = *(_QWORD *)(a2 + 32);
        v35 = *(_QWORD *)(a1 + 32);
        a4 = v34 + 40LL * *(unsigned int *)(a2 + 40);
        if ( v34 == a4 )
        {
          *(_DWORD *)(a1 + 40) = v11;
        }
        else
        {
          do
          {
            if ( v35 )
            {
              *(_QWORD *)v35 = *(_QWORD *)v34;
              *(_DWORD *)(v35 + 16) = *(_DWORD *)(v34 + 16);
              *(_QWORD *)(v35 + 8) = *(_QWORD *)(v34 + 8);
              v36 = *(_DWORD *)(v34 + 32);
              *(_DWORD *)(v34 + 16) = 0;
              *(_DWORD *)(v35 + 32) = v36;
              *(_QWORD *)(v35 + 24) = *(_QWORD *)(v34 + 24);
              *(_DWORD *)(v34 + 32) = 0;
            }
            v34 += 40;
            v35 += 40;
          }
          while ( a4 != v34 );
          v37 = *(unsigned int *)(a2 + 40);
          v38 = *(_QWORD *)(a2 + 32);
          *(_DWORD *)(a1 + 40) = v11;
          v39 = v38 + 40 * v37;
          while ( v39 != v38 )
          {
            v39 -= 40;
            if ( *(_DWORD *)(v39 + 32) > 0x40u )
            {
              v40 = *(_QWORD *)(v39 + 24);
              if ( v40 )
                j_j___libc_free_0_0(v40);
            }
            if ( *(_DWORD *)(v39 + 16) > 0x40u )
            {
              v41 = *(_QWORD *)(v39 + 8);
              if ( v41 )
                j_j___libc_free_0_0(v41);
            }
          }
        }
        *(_DWORD *)(a2 + 40) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v32;
        v42 = *(_DWORD *)(a2 + 44);
        *(_DWORD *)(a1 + 40) = v11;
        *(_DWORD *)(a1 + 44) = v42;
        *(_QWORD *)(a2 + 32) = v33;
        *(_QWORD *)(a2 + 40) = 0;
      }
    }
  }
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  v12 = *(_DWORD *)(a2 + 72);
  v13 = *(_QWORD *)(a2 + 56);
  ++*(_QWORD *)(a2 + 48);
  *(_DWORD *)(a1 + 72) = v12;
  *(_QWORD *)(a1 + 56) = v13;
  v14 = *(_QWORD *)(a2 + 64);
  *(_QWORD *)(a2 + 56) = 0;
  *(_QWORD *)(a2 + 64) = 0;
  *(_DWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 48) = 1;
  *(_QWORD *)(a1 + 64) = v14;
  *(_QWORD *)(a1 + 88) = 0x800000000LL;
  if ( *(_DWORD *)(a2 + 88) )
    sub_2303B80(a1 + 80, (char **)(a2 + 80), v14, a4, a5, a6);
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = a1 + 168;
  *(_QWORD *)(a1 + 192) = a1 + 168;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  v46 = (_QWORD *)(a1 + 160);
  sub_23082A0(0);
  v17 = *(_QWORD *)(a2 + 184);
  *(_QWORD *)(a1 + 184) = a1 + 168;
  *(_QWORD *)(a1 + 192) = a1 + 168;
  *(_QWORD *)(a1 + 176) = 0;
  for ( *(_QWORD *)(a1 + 200) = 0; a2 + 168 != v17; v17 = sub_220EF30(v17) )
  {
    if ( (*(_BYTE *)(v17 + 40) & 1) != 0 )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v48 = (__int64)&v48;
      v49 = 1;
      v50 = v18;
      v19 = sub_2365AE0(v46, (__int64)&v48);
      v20 = v19 + 4;
      if ( (v19[5] & 1) == 0 )
        v20 = 0;
      for ( i = *(_QWORD *)(v17 + 40) & 0xFFFFFFFFFFFFFFFELL; i; i = *(_QWORD *)(i + 8) & 0xFFFFFFFFFFFFFFFELL )
      {
        v22 = *(_QWORD *)(i + 16);
        v47 = v20;
        v48 = (__int64)&v48;
        v49 = 1;
        v50 = v22;
        v23 = sub_2365AE0(v46, (__int64)&v48);
        v20 = v47;
        v24 = v23;
        v25 = (unsigned __int64)(v23 + 4);
        if ( (v24[5] & 1) == 0 )
          v25 = 0;
        if ( v47 != (_QWORD *)v25 )
        {
          *(_QWORD *)(*v47 + 8LL) = v25 | *(_QWORD *)(*v47 + 8LL) & 1LL;
          *v47 = *(_QWORD *)v25;
          *(_QWORD *)(v25 + 8) &= ~1uLL;
          *(_QWORD *)v25 = v47;
        }
      }
    }
  }
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  v26 = *(_QWORD *)(a2 + 216);
  v27 = *(_DWORD *)(a2 + 232);
  ++*(_QWORD *)(a2 + 208);
  *(_QWORD *)(a1 + 216) = v26;
  v28 = *(_QWORD *)(a2 + 224);
  *(_QWORD *)(a2 + 216) = 0;
  *(_QWORD *)(a2 + 224) = 0;
  *(_DWORD *)(a2 + 232) = 0;
  *(_QWORD *)(a1 + 208) = 1;
  *(_QWORD *)(a1 + 224) = v28;
  *(_DWORD *)(a1 + 232) = v27;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 248) = 0;
  v29 = *(_DWORD *)(a2 + 248);
  if ( v29 && a1 + 240 != a2 + 240 )
  {
    v43 = *(_QWORD *)(a2 + 240);
    if ( v43 == a2 + 256 )
    {
      sub_C8D5F0(a1 + 240, (const void *)(a1 + 256), v29, 0x10u, v15, v16);
      v44 = 16LL * *(unsigned int *)(a2 + 248);
      if ( v44 )
        memcpy(*(void **)(a1 + 240), *(const void **)(a2 + 240), v44);
      *(_DWORD *)(a1 + 248) = v29;
      *(_DWORD *)(a2 + 248) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 240) = v43;
      v45 = *(_DWORD *)(a2 + 252);
      *(_DWORD *)(a1 + 248) = v29;
      *(_DWORD *)(a1 + 252) = v45;
      *(_QWORD *)(a2 + 240) = a2 + 256;
      *(_QWORD *)(a2 + 248) = 0;
    }
  }
  result = *(_QWORD *)(a2 + 256);
  *(_QWORD *)(a1 + 256) = result;
  return result;
}
