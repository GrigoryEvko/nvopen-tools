// Function: sub_1ED9CC0
// Address: 0x1ed9cc0
//
void __fastcall sub_1ED9CC0(__int64 *a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int64 v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // edi
  unsigned int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // esi
  unsigned int v20; // edi
  unsigned __int64 v21; // [rsp+8h] [rbp-B8h]
  char v22; // [rsp+10h] [rbp-B0h]
  __int64 v23; // [rsp+10h] [rbp-B0h]
  __int64 v25; // [rsp+20h] [rbp-A0h]
  __int64 v27; // [rsp+30h] [rbp-90h]
  __int64 *v29; // [rsp+40h] [rbp-80h] BYREF
  __int64 v30; // [rsp+48h] [rbp-78h]
  _BYTE v31[112]; // [rsp+50h] [rbp-70h] BYREF

  v3 = *(unsigned int *)(*a1 + 72);
  if ( (_DWORD)v3 )
  {
    v22 = 0;
    v4 = 0;
    v27 = 8 * v3;
    while ( 1 )
    {
      while ( 1 )
      {
        v5 = a1[14] + 5 * v4;
        if ( *(_DWORD *)v5 == 1 || !*(_DWORD *)v5 && *(_BYTE *)(v5 + 32) && *(_BYTE *)(v5 + 33) )
        {
          v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 64) + v4) + 8LL);
          v7 = 0;
          if ( *(_BYTE *)(v5 + 35) )
            v7 = *(_QWORD *)(*(_QWORD *)(v5 + 24) + 8LL);
          v8 = *(_QWORD *)(a2 + 104);
          if ( v8 )
            break;
        }
        v4 += 8;
        if ( v4 == v27 )
        {
LABEL_22:
          if ( v22 )
            sub_1DB4C70(a2);
          return;
        }
      }
      v25 = a1[14] + 5 * v4;
      v21 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      do
      {
        v9 = v6 & 0xFFFFFFFFFFFFFFF8LL;
        v10 = (__int64 *)sub_1DB3C70((__int64 *)v8, v6 & 0xFFFFFFFFFFFFFFF8LL);
        v11 = *(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8);
        if ( v10 == (__int64 *)v11 )
          goto LABEL_20;
        v12 = *(_DWORD *)(v9 + 24);
        v13 = *(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        if ( (unsigned __int64)(v13 | (*v10 >> 1) & 3) > v12 )
        {
          v14 = 0;
        }
        else
        {
          v14 = v10[2];
          if ( v9 == (v10[1] & 0xFFFFFFFFFFFFFFF8LL) )
          {
            if ( (__int64 *)v11 == v10 + 3 )
              goto LABEL_18;
            v13 = *(_DWORD *)((v10[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
            v10 += 3;
          }
          if ( v9 == *(_QWORD *)(v14 + 8) )
            v14 = 0;
        }
        if ( v12 < v13 || (v15 = v10[2], v16 = v10[1], !v15) )
        {
LABEL_18:
          if ( !v14 )
            goto LABEL_20;
LABEL_19:
          *a3 |= *(_DWORD *)(v8 + 112);
          goto LABEL_20;
        }
        if ( !v14 )
        {
          v23 = v15;
          v30 = 0x800000000LL;
          v29 = (__int64 *)v31;
          sub_1DC0B50(a1[5], v8, v6, (__int64)&v29);
          *(_QWORD *)(v23 + 8) = 0;
          if ( *(_BYTE *)(v25 + 35) )
          {
            v17 = (_QWORD *)sub_1DB3C70((__int64 *)v8, v21);
            v18 = *(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8);
            if ( v17 != (_QWORD *)v18 )
            {
              v19 = *(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24);
              v20 = *(_DWORD *)(v21 + 24);
              if ( (unsigned __int64)(v19 | ((__int64)*v17 >> 1) & 3) > v20 || v21 != (v17[1] & 0xFFFFFFFFFFFFFFF8LL) )
                goto LABEL_38;
              if ( (_QWORD *)v18 != v17 + 3 )
              {
                v19 = *(_DWORD *)((v17[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
                v17 += 3;
LABEL_38:
                if ( v20 >= v19 && ((*((_BYTE *)v17 + 8) ^ 6) & 6) != 0 && v17[2] )
                  sub_1DBC0D0((_QWORD *)a1[5], v8, v29, (unsigned int)v30, 0, 0);
              }
            }
          }
          if ( v29 != (__int64 *)v31 )
            _libc_free((unsigned __int64)v29);
          v22 = 1;
          goto LABEL_20;
        }
        if ( (((unsigned __int8)v16 ^ 6) & 6) == 0 )
          goto LABEL_19;
LABEL_20:
        v8 = *(_QWORD *)(v8 + 104);
      }
      while ( v8 );
      v4 += 8;
      if ( v4 == v27 )
        goto LABEL_22;
    }
  }
}
