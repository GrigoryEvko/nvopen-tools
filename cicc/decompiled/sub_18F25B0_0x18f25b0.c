// Function: sub_18F25B0
// Address: 0x18f25b0
//
__int64 __fastcall sub_18F25B0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r12
  int v8; // edi
  _QWORD *v9; // r10
  int v10; // edi
  __int64 v11; // r14
  bool v12; // zf
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rax
  _BYTE *v17; // rdi
  int v18; // ebx
  int v19; // ecx
  _QWORD *v20; // rsi
  unsigned int v21; // eax
  __int64 *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // r12
  int v25; // r8d
  int v26; // edx
  int v27; // r8d
  unsigned __int8 v28; // [rsp+1Fh] [rbp-151h]
  __int64 v29; // [rsp+20h] [rbp-150h] BYREF
  __int64 v30; // [rsp+28h] [rbp-148h]
  _QWORD *v31; // [rsp+30h] [rbp-140h] BYREF
  int v32; // [rsp+38h] [rbp-138h]
  _BYTE *v33; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+B8h] [rbp-B8h]
  _BYTE v35[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v2 = &v31;
  v29 = 0;
  v30 = 1;
  do
    *v2++ = -8;
  while ( v2 != &v33 );
  v3 = *(_QWORD *)(a1 + 80);
  v4 = a1 + 72;
  v33 = v35;
  v34 = 0x1000000000LL;
  if ( a1 + 72 == v3 )
  {
LABEL_9:
    v28 = 0;
  }
  else
  {
    if ( !v3 )
      BUG();
    while ( 1 )
    {
      v5 = *(_QWORD *)(v3 + 24);
      if ( v5 != v3 + 16 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 == v3 )
        goto LABEL_9;
      if ( !v3 )
        BUG();
    }
    v28 = 0;
    while ( v3 != v4 )
    {
      v8 = 16;
      v9 = &v31;
      if ( (v30 & 1) == 0 )
      {
        v9 = v31;
        v8 = v32;
      }
      v10 = v8 - 1;
      while ( 1 )
      {
        v11 = v5 - 24;
        v12 = v5 == 0;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v12 )
          v11 = 0;
        while ( 1 )
        {
          v13 = v3 - 24;
          if ( !v3 )
            v13 = 0;
          if ( v5 != v13 + 40 )
            break;
          v3 = *(_QWORD *)(v3 + 8);
          if ( v4 == v3 )
            break;
          if ( !v3 )
            BUG();
          v5 = *(_QWORD *)(v3 + 24);
        }
        if ( (v30 & 1) == 0 && !v32 )
          break;
        v14 = v10 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v15 = v9[v14];
        if ( v11 != v15 )
        {
          v25 = 1;
          while ( v15 != -8 )
          {
            v14 = v10 & (v25 + v14);
            v15 = v9[v14];
            if ( v11 == v15 )
              goto LABEL_29;
            ++v25;
          }
          break;
        }
LABEL_29:
        if ( v4 == v3 )
          goto LABEL_30;
      }
      if ( (unsigned __int8)sub_1AE9990(v11, a2) )
        v28 |= sub_18F21F0(v11, (__int64)&v29, a2);
    }
LABEL_30:
    v16 = (unsigned int)v34;
    v17 = v33;
    if ( (_DWORD)v34 )
    {
      v18 = v28;
      while ( 1 )
      {
        v24 = *(_QWORD *)&v17[8 * v16 - 8];
        if ( (v30 & 1) != 0 )
          break;
        v20 = v31;
        v19 = v32 - 1;
        if ( v32 )
          goto LABEL_33;
LABEL_35:
        LODWORD(v34) = v34 - 1;
        if ( (unsigned __int8)sub_1AE9990(v24, a2) )
          v18 |= sub_18F21F0(v24, (__int64)&v29, a2);
        v16 = (unsigned int)v34;
        v17 = v33;
        if ( !(_DWORD)v34 )
        {
          v28 = v18;
          goto LABEL_46;
        }
      }
      v19 = 15;
      v20 = &v31;
LABEL_33:
      v21 = v19 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v22 = &v20[v21];
      v23 = *v22;
      if ( v24 == *v22 )
      {
LABEL_34:
        *v22 = -16;
        ++HIDWORD(v30);
        LODWORD(v30) = (2 * ((unsigned int)v30 >> 1) - 2) | v30 & 1;
      }
      else
      {
        v26 = 1;
        while ( v23 != -8 )
        {
          v27 = v26 + 1;
          v21 = v19 & (v26 + v21);
          v22 = &v20[v21];
          v23 = *v22;
          if ( v24 == *v22 )
            goto LABEL_34;
          v26 = v27;
        }
      }
      goto LABEL_35;
    }
LABEL_46:
    if ( v17 != v35 )
      _libc_free((unsigned __int64)v17);
  }
  if ( (v30 & 1) == 0 )
    j___libc_free_0(v31);
  return v28;
}
