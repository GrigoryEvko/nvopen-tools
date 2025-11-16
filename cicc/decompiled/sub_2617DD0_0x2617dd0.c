// Function: sub_2617DD0
// Address: 0x2617dd0
//
__int64 __fastcall sub_2617DD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r14d
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // r8
  char v11; // al
  __int64 v12; // r10
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v16; // rax
  _BYTE *v17; // rsi
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-98h]
  __int64 v21; // [rsp+8h] [rbp-98h]
  __int64 v22; // [rsp+10h] [rbp-90h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  unsigned __int64 v24; // [rsp+18h] [rbp-88h]
  _BYTE *v25; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+28h] [rbp-78h]
  _BYTE v27[112]; // [rsp+30h] [rbp-70h] BYREF

  v24 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 != a2 + 24 && *(_DWORD *)a1 )
  {
    v2 = *(_QWORD *)(a2 + 32);
    v3 = 0;
    while ( 1 )
    {
      v4 = 0;
      if ( v2 )
        v4 = v2 - 56;
      v5 = v4;
      if ( (unsigned __int8)sub_B2D610(v4, 48) )
        goto LABEL_5;
      if ( v5 + 72 == (*(_QWORD *)(v5 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_5;
      v6 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 24))(*(_QWORD *)(a1 + 32), v5);
      if ( *(_QWORD *)(v6 + 40) == *(_QWORD *)(v6 + 32) )
        goto LABEL_5;
      v7 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 8))(*(_QWORD *)(a1 + 16), v5);
      v8 = *(__int64 **)(v6 + 32);
      v9 = *(_QWORD *)(v6 + 40);
      v10 = v7;
      if ( (__int64 *)v9 != v8 + 1 )
        goto LABEL_4;
      v20 = v7;
      v22 = *v8;
      v11 = sub_D4B3D0(*v8);
      v12 = v22;
      v10 = v20;
      if ( !v11 )
        goto LABEL_33;
      v13 = *(_QWORD *)(v5 + 80);
      if ( !v13 )
        BUG();
      v14 = *(_QWORD *)(v13 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v14 == v13 + 24 )
        goto LABEL_38;
      if ( !v14 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA )
LABEL_38:
        BUG();
      if ( *(_BYTE *)(v14 - 24) == 31 && (*(_DWORD *)(v14 - 20) & 0x7FFFFFF) == 1 )
      {
        v21 = v22;
        v23 = v10;
        v16 = sub_B46EC0(v14 - 24, 0);
        v12 = v21;
        LODWORD(v10) = v23;
        if ( **(_QWORD **)(v21 + 32) == v16 )
        {
          v25 = v27;
          v26 = 0x800000000LL;
          sub_D472F0(v21, (__int64)&v25);
          v12 = v21;
          v17 = &v25[8 * (unsigned int)v26];
          v10 = v23;
          if ( v25 == v17 )
          {
LABEL_31:
            if ( v25 != v27 )
            {
              _libc_free((unsigned __int64)v25);
              v10 = v23;
              v12 = v21;
            }
LABEL_33:
            v9 = *(_QWORD *)(v12 + 16);
            v8 = *(__int64 **)(v12 + 8);
LABEL_4:
            v3 |= sub_2617C90((_DWORD *)a1, v8, v9, v6, v10);
LABEL_5:
            if ( !*(_DWORD *)a1 )
              return v3;
            goto LABEL_6;
          }
          v18 = (unsigned __int64)v25;
          while ( 1 )
          {
            v19 = *(_QWORD *)(*(_QWORD *)v18 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v19 == *(_QWORD *)v18 + 48LL )
              goto LABEL_41;
            if ( !v19 )
              BUG();
            if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
LABEL_41:
              BUG();
            if ( *(_BYTE *)(v19 - 24) != 30 )
              break;
            v18 += 8LL;
            if ( v17 == (_BYTE *)v18 )
              goto LABEL_31;
          }
          if ( v25 != v27 )
          {
            _libc_free((unsigned __int64)v25);
            v12 = v21;
            LODWORD(v10) = v23;
          }
        }
      }
      v3 |= sub_2617A30(a1, v12, v6, v10);
      if ( !*(_DWORD *)a1 )
        return v3;
LABEL_6:
      if ( v2 == v24 )
        return v3;
      v2 = *(_QWORD *)(v2 + 8);
    }
  }
  return 0;
}
