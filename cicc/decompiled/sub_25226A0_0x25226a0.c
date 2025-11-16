// Function: sub_25226A0
// Address: 0x25226a0
//
__int64 __fastcall sub_25226A0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  _QWORD *v6; // r14
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // edx
  int v11; // ecx
  unsigned int v12; // edx
  __int64 v13; // rdi
  unsigned int v14; // r15d
  unsigned int v15; // eax
  __int64 *v17; // rdx
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 v21; // r12
  unsigned __int8 *v22; // rdx
  int v23; // eax
  unsigned __int64 v24; // rax
  int v25; // r8d
  __int64 v26; // rbx
  void *v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  _BYTE *v30; // r14
  unsigned __int64 v31; // rax
  __int64 v32; // [rsp+10h] [rbp-50h]
  unsigned int v33; // [rsp+18h] [rbp-48h]
  unsigned int v34; // [rsp+1Ch] [rbp-44h]
  __int64 v35[8]; // [rsp+20h] [rbp-40h] BYREF

  v1 = 30;
  v3 = sub_C996C0("Attributor::manifestAttributes", 30, 0, 0);
  v4 = *(_QWORD **)(a1 + 256);
  v34 = 1;
  v32 = v3;
  v5 = &v4[*(unsigned int *)(a1 + 264)];
  v33 = *(_DWORD *)(a1 + 264);
  if ( v4 == v5 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      v6 = (_QWORD *)(*v4 & 0xFFFFFFFFFFFFFFF8LL);
      v7 = (*(__int64 (__fastcall **)(_QWORD *))(*v6 + 40LL))(v6);
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v7 + 24LL))(v7) )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 32LL))(v7);
      if ( v6[10] || !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v7 + 16LL))(v7) )
        goto LABEL_3;
      if ( !sub_2509740(v6 + 9) )
        break;
      v8 = sub_25096F0(v6 + 9);
      v9 = *(_QWORD *)(a1 + 200);
      if ( !*(_DWORD *)(v9 + 40) )
        break;
      v1 = *(_QWORD *)(v9 + 8);
      v10 = *(_DWORD *)(v9 + 24);
      if ( v10 )
      {
        v11 = v10 - 1;
        v12 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v13 = *(_QWORD *)(v1 + 8LL * v12);
        if ( v8 != v13 )
        {
          v25 = 1;
          while ( v13 != -4096 )
          {
            v12 = v11 & (v25 + v12);
            v13 = *(_QWORD *)(v1 + 8LL * v12);
            if ( v8 == v13 )
              goto LABEL_12;
            ++v25;
          }
          goto LABEL_3;
        }
        break;
      }
LABEL_3:
      if ( v5 == ++v4 )
        goto LABEL_15;
    }
LABEL_12:
    v1 = (__int64)v6;
    LOBYTE(v35[0]) = 0;
    if ( (unsigned __int8)sub_251C440(a1, (__int64)v6, 0, v35, 1, 1) )
      goto LABEL_3;
    v14 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*v6 + 88LL))(v6, a1);
    if ( !v14 && (unsigned __int8)sub_C92250() )
      (*(void (__fastcall **)(_QWORD *))(*v6 + 96LL))(v6);
    v1 = v14;
    ++v4;
    v34 = sub_250C0B0(v34, v14);
  }
  while ( v5 != v4 );
LABEL_15:
  v15 = *(_DWORD *)(a1 + 264);
  if ( v15 != v33 )
  {
    v26 = *(_QWORD *)(a1 + 256);
    if ( v33 )
      v26 += 8LL * v33;
    if ( v15 > v33 )
    {
      do
      {
        v27 = sub_CB72A0();
        v26 += 8;
        v28 = sub_904010((__int64)v27, "Unexpected abstract attribute: ");
        v29 = sub_CB5A80(v28, *(_QWORD *)(v26 - 8) & 0xFFFFFFFFFFFFFFF8LL);
        v30 = (_BYTE *)sub_904010(v29, " :: ");
        v31 = sub_250D070((_QWORD *)((*(_QWORD *)(v26 - 8) & 0xFFFFFFFFFFFFFFF8LL) + 72));
        sub_A69870(v31, v30, 0);
        sub_904010((__int64)v30, "\n");
        ++v33;
      }
      while ( *(_DWORD *)(a1 + 264) > v33 );
    }
    BUG();
  }
LABEL_16:
  if ( *(_DWORD *)(a1 + 16) )
  {
    v17 = *(__int64 **)(a1 + 8);
    v18 = &v17[2 * *(unsigned int *)(a1 + 24)];
    if ( v17 != v18 )
    {
      while ( 1 )
      {
        v19 = *v17;
        v20 = v17;
        if ( *v17 != -4096 && v19 != -8192 )
          break;
        v17 += 2;
        if ( v18 == v17 )
          goto LABEL_17;
      }
      if ( v17 != v18 )
      {
        v21 = 0x8000000000041LL;
        do
        {
          v35[1] = 0;
          v35[0] = v19 & 0xFFFFFFFFFFFFFFFCLL;
          nullsub_1518();
          v22 = (unsigned __int8 *)(v35[0] & 0xFFFFFFFFFFFFFFFCLL);
          if ( (v35[0] & 3) == 3 )
            v22 = (unsigned __int8 *)*((_QWORD *)v22 + 3);
          v23 = *v22;
          if ( (unsigned __int8)v23 > 0x1Cu
            && (v24 = (unsigned int)(v23 - 34), (unsigned __int8)v24 <= 0x33u)
            && _bittest64(&v21, v24) )
          {
            *((_QWORD *)v22 + 9) = v20[1];
          }
          else
          {
            *((_QWORD *)sub_250CBE0(v35, v1) + 15) = v20[1];
          }
          v20 += 2;
          if ( v20 == v18 )
            break;
          while ( 1 )
          {
            v19 = *v20;
            if ( *v20 != -8192 && v19 != -4096 )
              break;
            v20 += 2;
            if ( v18 == v20 )
              goto LABEL_17;
          }
        }
        while ( v18 != v20 );
      }
    }
  }
LABEL_17:
  if ( v32 )
    sub_C9AF60(v32);
  return v34;
}
