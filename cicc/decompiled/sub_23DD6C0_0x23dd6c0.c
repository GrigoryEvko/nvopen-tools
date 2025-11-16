// Function: sub_23DD6C0
// Address: 0x23dd6c0
//
__int64 __fastcall sub_23DD6C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r15
  int v11; // edx
  unsigned int v12; // ecx
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  int v15; // r15d
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // rbx
  int v21; // eax
  int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rbx
  int v28; // eax
  int v29; // eax
  unsigned int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 *v35; // rax
  _BYTE v38[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v39; // [rsp+30h] [rbp-70h]
  const char *v40; // [rsp+40h] [rbp-60h] BYREF
  __int16 v41; // [rsp+60h] [rbp-40h]

  v39 = 257;
  v41 = 257;
  v8 = sub_BD2DA0(80);
  v9 = v8;
  if ( v8 )
  {
    sub_B44260(v8, a1, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v9 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v9, &v40);
    sub_BD2A10(v9, *(_DWORD *)(v9 + 72), 1);
  }
  if ( *(_BYTE *)v9 > 0x1Cu )
  {
    switch ( *(_BYTE *)v9 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_8;
      case 'T':
      case 'U':
      case 'V':
        v10 = *(_QWORD *)(v9 + 8);
        v11 = *(unsigned __int8 *)(v10 + 8);
        v12 = v11 - 17;
        v13 = *(_BYTE *)(v10 + 8);
        if ( (unsigned int)(v11 - 17) <= 1 )
          v13 = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
        if ( v13 <= 3u || v13 == 5 || (v13 & 0xFD) == 4 )
          goto LABEL_8;
        if ( (_BYTE)v11 == 15 )
        {
          if ( (*(_BYTE *)(v10 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v9 + 8)) )
            break;
          v35 = *(__int64 **)(v10 + 16);
          v10 = *v35;
          v11 = *(unsigned __int8 *)(*v35 + 8);
          v12 = v11 - 17;
        }
        else if ( (_BYTE)v11 == 16 )
        {
          do
          {
            v10 = *(_QWORD *)(v10 + 24);
            LOBYTE(v11) = *(_BYTE *)(v10 + 8);
          }
          while ( (_BYTE)v11 == 16 );
          v12 = (unsigned __int8)v11 - 17;
        }
        if ( v12 <= 1 )
          LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
        if ( (unsigned __int8)v11 <= 3u || (_BYTE)v11 == 5 || (v11 & 0xFD) == 4 )
        {
LABEL_8:
          v14 = a2[12];
          v15 = *((_DWORD *)a2 + 26);
          if ( v14 )
            sub_B99FD0(v9, 3u, v14);
          sub_B45150(v9, v15);
        }
        break;
      default:
        break;
    }
  }
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
    a2[11],
    v9,
    v38,
    a2[7],
    a2[8]);
  v16 = *a2;
  v17 = *a2 + 16LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v17 )
  {
    do
    {
      v18 = *(_QWORD *)(v16 + 8);
      v19 = *(_DWORD *)v16;
      v16 += 16;
      sub_B99FD0(v9, v19, v18);
    }
    while ( v17 != v16 );
  }
  v20 = *(_QWORD *)(a3 + 40);
  v21 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
  if ( v21 == *(_DWORD *)(v9 + 72) )
  {
    sub_B48D90(v9);
    v21 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
  }
  v22 = (v21 + 1) & 0x7FFFFFF;
  v23 = v22 | *(_DWORD *)(v9 + 4) & 0xF8000000;
  v24 = *(_QWORD *)(v9 - 8) + 32LL * (unsigned int)(v22 - 1);
  *(_DWORD *)(v9 + 4) = v23;
  if ( *(_QWORD *)v24 )
  {
    v25 = *(_QWORD *)(v24 + 8);
    **(_QWORD **)(v24 + 16) = v25;
    if ( v25 )
      *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
  }
  *(_QWORD *)v24 = a6;
  if ( a6 )
  {
    v26 = *(_QWORD *)(a6 + 16);
    *(_QWORD *)(v24 + 8) = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = v24 + 8;
    *(_QWORD *)(v24 + 16) = a6 + 16;
    *(_QWORD *)(a6 + 16) = v24;
  }
  *(_QWORD *)(*(_QWORD *)(v9 - 8) + 32LL * *(unsigned int *)(v9 + 72) + 8LL * ((*(_DWORD *)(v9 + 4) & 0x7FFFFFFu) - 1)) = v20;
  v27 = *(_QWORD *)(a5 + 40);
  v28 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
  if ( v28 == *(_DWORD *)(v9 + 72) )
  {
    sub_B48D90(v9);
    v28 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
  }
  v29 = (v28 + 1) & 0x7FFFFFF;
  v30 = v29 | *(_DWORD *)(v9 + 4) & 0xF8000000;
  v31 = *(_QWORD *)(v9 - 8) + 32LL * (unsigned int)(v29 - 1);
  *(_DWORD *)(v9 + 4) = v30;
  if ( *(_QWORD *)v31 )
  {
    v32 = *(_QWORD *)(v31 + 8);
    **(_QWORD **)(v31 + 16) = v32;
    if ( v32 )
      *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 16);
  }
  *(_QWORD *)v31 = a4;
  if ( a4 )
  {
    v33 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(v31 + 8) = v33;
    if ( v33 )
      *(_QWORD *)(v33 + 16) = v31 + 8;
    *(_QWORD *)(v31 + 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = v31;
  }
  *(_QWORD *)(*(_QWORD *)(v9 - 8) + 32LL * *(unsigned int *)(v9 + 72) + 8LL * ((*(_DWORD *)(v9 + 4) & 0x7FFFFFFu) - 1)) = v27;
  return v9;
}
