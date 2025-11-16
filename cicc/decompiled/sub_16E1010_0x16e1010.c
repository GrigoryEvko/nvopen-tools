// Function: sub_16E1010
// Address: 0x16e1010
//
void __fastcall sub_16E1010(__int64 a1, __int64 a2)
{
  char *v2; // rax
  char *v3; // rax
  int v4; // r9d
  int v5; // eax
  unsigned __int64 v6; // r13
  int v7; // eax
  unsigned int v8; // r14d
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int v11; // r8d
  char *v12[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v13; // [rsp+10h] [rbp-70h] BYREF
  __int64 v14; // [rsp+18h] [rbp-68h]
  _BYTE v15[96]; // [rsp+20h] [rbp-60h] BYREF

  sub_16E2FC0(a1, a2);
  v14 = 0x400000000LL;
  v2 = *(char **)a1;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  v12[0] = v2;
  v3 = *(char **)(a1 + 8);
  v13 = v15;
  v12[1] = v3;
  sub_16D2880(v12, (__int64)&v13, 45, 3, 1, v4);
  if ( (_DWORD)v14
    && (*(_DWORD *)(a1 + 32) = sub_16DF2D0(*(_QWORD *)v13, *((_QWORD *)v13 + 1)),
        v7 = sub_16DE070(*(_QWORD *)v13, *((_QWORD *)v13 + 1)),
        v8 = v14,
        *(_DWORD *)(a1 + 36) = v7,
        v8 > 1) )
  {
    v6 = (unsigned __int64)v13;
    *(_DWORD *)(a1 + 40) = sub_16DE1A0(*((_QWORD *)v13 + 2), *((_QWORD *)v13 + 3));
    if ( v8 == 2
      || (*(_DWORD *)(a1 + 44) = sub_16DE880(*(_QWORD *)(v6 + 32), *(_QWORD *)(v6 + 40), v9, v10, v11), v8 == 3) )
    {
      v5 = *(_DWORD *)(a1 + 52);
    }
    else
    {
      *(_DWORD *)(a1 + 48) = sub_16DE390(*(_QWORD *)(v6 + 48), *(_QWORD *)(v6 + 56));
      v5 = sub_16DDFE0(*(_QWORD *)(v6 + 48), *(_QWORD *)(v6 + 56));
      *(_DWORD *)(a1 + 52) = v5;
    }
  }
  else
  {
    v5 = *(_DWORD *)(a1 + 52);
    v6 = (unsigned __int64)v13;
  }
  if ( !v5 )
    *(_DWORD *)(a1 + 52) = sub_16DE120(a1);
  if ( (_BYTE *)v6 != v15 )
    _libc_free(v6);
}
