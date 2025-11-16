// Function: sub_23CA790
// Address: 0x23ca790
//
void __fastcall sub_23CA790(__int64 a1)
{
  unsigned __int64 v1; // rbx
  __int64 v2; // rax
  _QWORD *v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // r13
  int v7; // esi
  int v8; // eax
  _DWORD *v9; // rax
  __int64 v10; // r9
  _DWORD *v11; // r15
  _DWORD *v12; // r14
  int v13; // eax
  __int64 v14; // r9
  unsigned __int64 v15; // r8
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 *v18; // rax
  __int64 v20; // [rsp+18h] [rbp-88h]
  unsigned __int64 v21; // [rsp+20h] [rbp-80h]
  _QWORD *v22; // [rsp+30h] [rbp-70h] BYREF
  __int64 v23; // [rsp+38h] [rbp-68h]
  _QWORD v24[12]; // [rsp+40h] [rbp-60h] BYREF

  v2 = *(_QWORD *)(a1 + 216);
  v3 = v24;
  v24[0] = v2;
  v23 = 0x300000001LL;
  v4 = 1;
  v22 = v24;
  v24[1] = 0;
  while ( v4 )
  {
    v5 = (__int64)&v3[2 * v4 - 2];
    v6 = *(_QWORD *)v5;
    v7 = *(_DWORD *)(v5 + 8);
    LODWORD(v23) = v4 - 1;
    sub_23CC700(v6);
    v8 = *(_DWORD *)(v6 + 8);
    if ( v8 != 1 )
      goto LABEL_6;
    if ( *(_DWORD *)(v6 + 56) )
    {
      v9 = *(_DWORD **)(v6 + 48);
      v10 = 4LL * *(unsigned int *)(v6 + 64);
      v11 = &v9[v10];
      if ( v9 != &v9[v10] )
      {
        while ( 1 )
        {
          v12 = v9;
          if ( *v9 <= 0xFFFFFFFD )
            break;
          v9 += 4;
          if ( v11 == v9 )
            goto LABEL_3;
        }
        if ( v11 != v9 )
        {
          do
          {
            v13 = sub_23CA740(*((_QWORD *)v12 + 1));
            v14 = *((_QWORD *)v12 + 1);
            v15 = (unsigned int)(v7 + v13) | v1 & 0xFFFFFFFF00000000LL;
            v16 = (unsigned int)v23;
            v1 = v15;
            v17 = (unsigned int)v23 + 1LL;
            if ( v17 > HIDWORD(v23) )
            {
              v20 = *((_QWORD *)v12 + 1);
              v21 = v15;
              sub_C8D5F0((__int64)&v22, v24, v17, 0x10u, v15, v14);
              v16 = (unsigned int)v23;
              v14 = v20;
              v15 = v21;
            }
            v12 += 4;
            v18 = &v22[2 * v16];
            *v18 = v14;
            v18[1] = v15;
            LODWORD(v23) = v23 + 1;
            if ( v12 == v11 )
              break;
            while ( *v12 > 0xFFFFFFFD )
            {
              v12 += 4;
              if ( v11 == v12 )
                goto LABEL_18;
            }
          }
          while ( v11 != v12 );
LABEL_18:
          v8 = *(_DWORD *)(v6 + 8);
LABEL_6:
          if ( !v8 )
            sub_23CC770(v6, (unsigned int)*(_QWORD *)(a1 + 8) - v7);
        }
      }
    }
LABEL_3:
    v4 = v23;
    v3 = v22;
  }
  if ( v3 != v24 )
    _libc_free((unsigned __int64)v3);
}
