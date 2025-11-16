// Function: sub_24781A0
// Address: 0x24781a0
//
void __fastcall sub_24781A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 *v3; // rdx
  __int64 v4; // rdx
  _BYTE *v5; // rax
  __int64 v6; // rax
  _BYTE *v7; // r11
  _DWORD **v8; // r9
  _DWORD *v9; // rdx
  int v10; // eax
  __int64 v11; // r8
  _DWORD *v12; // rcx
  __int64 v13; // rax
  int v14; // [rsp+4h] [rbp-15Ch]
  _BYTE *v15; // [rsp+8h] [rbp-158h]
  _DWORD **v16; // [rsp+10h] [rbp-150h]
  _BYTE *v17; // [rsp+18h] [rbp-148h]
  _BYTE v18[32]; // [rsp+20h] [rbp-140h] BYREF
  __int16 v19; // [rsp+40h] [rbp-120h]
  _DWORD *v20; // [rsp+50h] [rbp-110h] BYREF
  __int64 v21; // [rsp+58h] [rbp-108h]
  _DWORD v22[4]; // [rsp+60h] [rbp-100h] BYREF
  __int16 v23; // [rsp+70h] [rbp-F0h]
  unsigned int *v24[2]; // [rsp+A0h] [rbp-C0h] BYREF
  char v25; // [rsp+B0h] [rbp-B0h] BYREF
  void *v26; // [rsp+120h] [rbp-40h]

  sub_23D0AB0((__int64)v24, a2, 0, 0, 0);
  v2 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL) + 32LL);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v17 = (_BYTE *)sub_246F3F0(a1, *v3);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v5 = (_BYTE *)sub_246F3F0(a1, *(_QWORD *)(v4 + 32));
  v23 = 257;
  v6 = sub_A82480(v24, v17, v5, (__int64)&v20);
  v20 = v22;
  v7 = (_BYTE *)v6;
  v22[0] = v2;
  v21 = 0x1000000001LL;
  if ( v2 <= 1 )
  {
    v11 = 1;
    v12 = v22;
  }
  else
  {
    v8 = &v20;
    v9 = v22;
    v10 = 1;
    v11 = 1;
    while ( 1 )
    {
      v9[v11] = v10++;
      v11 = (unsigned int)(v21 + 1);
      LODWORD(v21) = v21 + 1;
      if ( v2 == v10 )
        break;
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v21) )
      {
        v14 = v10;
        v15 = v7;
        v16 = v8;
        sub_C8D5F0((__int64)v8, v22, v11 + 1, 4u, v11, (__int64)v8);
        v11 = (unsigned int)v21;
        v10 = v14;
        v7 = v15;
        v8 = v16;
      }
      v9 = v20;
    }
    v12 = v20;
  }
  v19 = 257;
  v13 = sub_A83CB0(v24, v17, v7, (__int64)v12, v11, (__int64)v18);
  sub_246EF60(a1, a2, v13);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
    sub_2477350(a1, a2);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  nullsub_61();
  v26 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v24[0] != &v25 )
    _libc_free((unsigned __int64)v24[0]);
}
