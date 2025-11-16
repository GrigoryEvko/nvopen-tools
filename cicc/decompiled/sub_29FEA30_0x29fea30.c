// Function: sub_29FEA30
// Address: 0x29fea30
//
__int64 __fastcall sub_29FEA30(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rbx
  int v12; // edx
  unsigned int v13; // ecx
  unsigned __int8 v14; // al
  __int64 *v15; // rax
  __int64 v16; // rdx
  int v17; // r12d
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rdx
  unsigned int v21; // esi
  _QWORD *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rdx
  unsigned int v26; // esi
  _QWORD v27[2]; // [rsp+0h] [rbp-90h] BYREF
  const char *v28; // [rsp+10h] [rbp-80h]
  __int64 v29; // [rsp+18h] [rbp-78h]
  __int16 v30; // [rsp+20h] [rbp-70h]
  _BYTE v31[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v32; // [rsp+50h] [rbp-40h]

  if ( *(_QWORD *)(*a1 + 80) == *(_QWORD *)(a2 + 8) )
    return a2;
  v2 = (__int64 *)a1[2];
  if ( !*(_BYTE *)a1[1] )
  {
    v28 = sub_BD5D20(a2);
    v3 = *a1;
    v30 = 1283;
    v27[0] = "wide.";
    v29 = v4;
    v5 = *(_QWORD *)(v3 + 80);
    if ( v5 != *(_QWORD *)(a2 + 8) )
    {
      v6 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v2[10] + 120LL))(
             v2[10],
             39,
             a2,
             *(_QWORD *)(v3 + 80));
      if ( !v6 )
      {
        v32 = 257;
        v22 = sub_BD2C40(72, 1u);
        v6 = (__int64)v22;
        if ( v22 )
          sub_B515B0((__int64)v22, a2, v5, (__int64)v31, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v2[11] + 16LL))(
          v2[11],
          v6,
          v27,
          v2[7],
          v2[8]);
        v23 = *v2;
        v24 = *v2 + 16LL * *((unsigned int *)v2 + 2);
        if ( *v2 != v24 )
        {
          do
          {
            v25 = *(_QWORD *)(v23 + 8);
            v26 = *(_DWORD *)v23;
            v23 += 16;
            sub_B99FD0(v6, v26, v25);
          }
          while ( v24 != v23 );
        }
      }
      return v6;
    }
    return a2;
  }
  v28 = sub_BD5D20(a2);
  v8 = *a1;
  v30 = 1283;
  v27[0] = "wide.";
  v29 = v9;
  v10 = *(_QWORD *)(v8 + 80);
  if ( v10 == *(_QWORD *)(a2 + 8) )
    return a2;
  v6 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v2[10] + 120LL))(
         v2[10],
         40,
         a2,
         *(_QWORD *)(v8 + 80));
  if ( !v6 )
  {
    v32 = 257;
    v6 = sub_B51D30(40, a2, v10, (__int64)v31, 0, 0);
    if ( *(_BYTE *)v6 > 0x1Cu )
    {
      switch ( *(_BYTE *)v6 )
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
          goto LABEL_25;
        case 'T':
        case 'U':
        case 'V':
          v11 = *(_QWORD *)(v6 + 8);
          v12 = *(unsigned __int8 *)(v11 + 8);
          v13 = v12 - 17;
          v14 = *(_BYTE *)(v11 + 8);
          if ( (unsigned int)(v12 - 17) <= 1 )
            v14 = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
          if ( v14 <= 3u || v14 == 5 || (v14 & 0xFD) == 4 )
            goto LABEL_25;
          if ( (_BYTE)v12 == 15 )
          {
            if ( (*(_BYTE *)(v11 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v6 + 8)) )
              break;
            v15 = *(__int64 **)(v11 + 16);
            v11 = *v15;
            v12 = *(unsigned __int8 *)(*v15 + 8);
            v13 = v12 - 17;
          }
          else if ( (_BYTE)v12 == 16 )
          {
            do
            {
              v11 = *(_QWORD *)(v11 + 24);
              LOBYTE(v12) = *(_BYTE *)(v11 + 8);
            }
            while ( (_BYTE)v12 == 16 );
            v13 = (unsigned __int8)v12 - 17;
          }
          if ( v13 <= 1 )
            LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
          if ( (unsigned __int8)v12 <= 3u || (_BYTE)v12 == 5 || (v12 & 0xFD) == 4 )
          {
LABEL_25:
            v16 = v2[12];
            v17 = *((_DWORD *)v2 + 26);
            if ( v16 )
              sub_B99FD0(v6, 3u, v16);
            sub_B45150(v6, v17);
          }
          break;
        default:
          break;
      }
    }
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v2[11] + 16LL))(
      v2[11],
      v6,
      v27,
      v2[7],
      v2[8]);
    v18 = *v2;
    v19 = *v2 + 16LL * *((unsigned int *)v2 + 2);
    if ( *v2 != v19 )
    {
      do
      {
        v20 = *(_QWORD *)(v18 + 8);
        v21 = *(_DWORD *)v18;
        v18 += 16;
        sub_B99FD0(v6, v21, v20);
      }
      while ( v19 != v18 );
    }
  }
  return v6;
}
