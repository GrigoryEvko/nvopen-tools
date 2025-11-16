// Function: sub_2A6E8C0
// Address: 0x2a6e8c0
//
void __fastcall sub_2A6E8C0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  _BYTE *v5; // rdi
  int v6; // r15d
  __int64 v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // rax
  _BYTE *v10; // [rsp+10h] [rbp-60h] BYREF
  __int64 v11; // [rsp+18h] [rbp-58h]
  __int64 v12; // [rsp+20h] [rbp-50h]
  _BYTE v13[72]; // [rsp+28h] [rbp-48h] BYREF

  v10 = v13;
  v11 = 0;
  v12 = 16;
  sub_2A68DD0(a1, a2, &v10);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = v10;
  if ( (_DWORD)v11 )
  {
    v6 = v11;
    v7 = 0;
    do
    {
      while ( !v5[v7] )
      {
        if ( v6 == ++v7 )
          goto LABEL_6;
      }
      v8 = v7++;
      v9 = sub_B46EC0(a2, v8);
      sub_2A6E670(a1, v4, v9);
      v5 = v10;
    }
    while ( v6 != v7 );
  }
LABEL_6:
  if ( v5 != v13 )
    _libc_free((unsigned __int64)v5);
}
