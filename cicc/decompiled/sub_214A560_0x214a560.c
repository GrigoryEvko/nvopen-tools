// Function: sub_214A560
// Address: 0x214a560
//
void __fastcall sub_214A560(__int64 a1)
{
  _BYTE **v2; // rbx
  _BYTE **v3; // r13
  _BYTE *v4; // rax
  __int64 v5; // rdi
  _QWORD *v6; // r13
  _QWORD *v7; // rbx
  _BYTE *v8; // [rsp+0h] [rbp-40h] BYREF
  __int16 v9; // [rsp+10h] [rbp-30h]

  v2 = *(_BYTE ***)(a1 + 16);
  v3 = &v2[4 * *(unsigned int *)(a1 + 24)];
  if ( v3 != v2 )
  {
    do
    {
      v4 = *v2;
      v5 = *(_QWORD *)(a1 + 8);
      v9 = 257;
      if ( *v4 )
      {
        v8 = v4;
        LOBYTE(v9) = 3;
      }
      v2 += 4;
      sub_38DD5A0(v5, &v8);
    }
    while ( v3 != v2 );
    v6 = *(_QWORD **)(a1 + 16);
    v7 = &v6[4 * *(unsigned int *)(a1 + 24)];
    while ( v6 != v7 )
    {
      while ( 1 )
      {
        v7 -= 4;
        if ( (_QWORD *)*v7 == v7 + 2 )
          break;
        j_j___libc_free_0(*v7, v7[2] + 1LL);
        if ( v6 == v7 )
          goto LABEL_9;
      }
    }
  }
LABEL_9:
  *(_DWORD *)(a1 + 24) = 0;
}
