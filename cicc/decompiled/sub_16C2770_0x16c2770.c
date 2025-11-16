// Function: sub_16C2770
// Address: 0x16c2770
//
__int64 __fastcall sub_16C2770(__int64 a1, unsigned int a2, __int64 a3)
{
  _BYTE *v3; // r12
  int *v4; // rax
  __int64 v5; // rcx
  int *v6; // rbx
  _BYTE *v7; // r12
  __int64 v8; // rdx
  ssize_t v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned int v12; // edx
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  int v17; // [rsp+0h] [rbp-4050h]
  _BYTE *v18; // [rsp+10h] [rbp-4040h] BYREF
  __int64 v19; // [rsp+18h] [rbp-4038h]
  _BYTE v20[16432]; // [rsp+20h] [rbp-4030h] BYREF

  v3 = v20;
  v18 = v20;
  v19 = 0x400000000000LL;
  v4 = __errno_location();
  v5 = 0;
  v6 = v4;
  while ( 1 )
  {
    v7 = &v3[v5];
    while ( 1 )
    {
      *v6 = 0;
      v9 = read(a2, v7, 0x4000u);
      if ( v9 != -1 )
        break;
      v8 = (unsigned int)*v6;
      if ( (_DWORD)v8 != 4 )
      {
        v17 = *v6;
        v14 = sub_2241E50(a2, v7, v8, v10, v11);
        *(_BYTE *)(a1 + 16) |= 1u;
        *(_DWORD *)a1 = v17;
        *(_QWORD *)(a1 + 8) = v14;
        goto LABEL_10;
      }
    }
    v12 = v9 + v19;
    LODWORD(v19) = v12;
    v5 = v12;
    if ( !v9 )
      break;
    v13 = v12 + 0x4000LL;
    if ( v13 > HIDWORD(v19) )
    {
      sub_16CD150(&v18, v20, v13, 1);
      v5 = (unsigned int)v19;
    }
    v3 = v18;
  }
  sub_16C26E0(a1, v18, v12, a3);
LABEL_10:
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  return a1;
}
