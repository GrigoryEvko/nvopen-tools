// Function: sub_1E0D590
// Address: 0x1e0d590
//
void __fastcall sub_1E0D590(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r15
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // r8
  int v14; // r9d
  bool v15; // cf
  __int64 *v16; // r14
  __int64 *v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  _BYTE *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-80h]
  _BYTE *v23; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+28h] [rbp-58h]
  _BYTE v25[80]; // [rsp+30h] [rbp-50h] BYREF

  v3 = *(_QWORD **)(a2 + 56);
  v4 = sub_15E38F0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL));
  v5 = sub_1649C60(v4);
  if ( !*(_BYTE *)(v5 + 16) )
    sub_1E2D790(v3[4], v5);
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
    sub_1E0CF70(v3, a2, v6, v7, v8, v9);
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
  {
    v10 = (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) - 1;
    while ( 1 )
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
        v11 = *(_QWORD *)(*(_QWORD *)(a1 - 8) + 24 * v10);
        if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 14 )
          goto LABEL_8;
      }
      else
      {
        v11 = *(_QWORD *)(a1 + 24 * (v10 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 14 )
        {
LABEL_8:
          v12 = sub_1649C60(v11);
          if ( *(_BYTE *)(v12 + 16) >= 4u )
            v12 = 0;
          v23 = (_BYTE *)v12;
          sub_1E0CED0(v3, a2, (__int64)&v23, 1, v13, v14);
          goto LABEL_11;
        }
      }
      v23 = v25;
      v24 = 0x400000000LL;
      if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
      {
        v16 = *(__int64 **)(v11 - 8);
        v17 = &v16[3 * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)];
        if ( v17 != v16 )
          goto LABEL_16;
      }
      else
      {
        v17 = (__int64 *)v11;
        v21 = 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
        v16 = (__int64 *)(v11 - v21);
        if ( v11 != v11 - v21 )
        {
          do
          {
LABEL_16:
            v9 = sub_1649C60(*v16);
            v18 = (unsigned int)v24;
            if ( (unsigned int)v24 >= HIDWORD(v24) )
            {
              v22 = v9;
              sub_16CD150((__int64)&v23, v25, 0, 8, (int)v8, v9);
              v18 = (unsigned int)v24;
              v9 = v22;
            }
            v16 += 3;
            *(_QWORD *)&v23[8 * v18] = v9;
            v19 = (unsigned int)(v24 + 1);
            LODWORD(v24) = v24 + 1;
          }
          while ( v16 != v17 );
          v20 = v23;
          goto LABEL_20;
        }
      }
      v20 = v25;
      v19 = 0;
LABEL_20:
      sub_1E0D450(v3, a2, (__int64)v20, v19, v8, v9);
      if ( v23 == v25 )
      {
LABEL_11:
        v15 = v10-- == 0;
        if ( v15 )
          return;
      }
      else
      {
        _libc_free((unsigned __int64)v23);
        v15 = v10-- == 0;
        if ( v15 )
          return;
      }
    }
  }
}
