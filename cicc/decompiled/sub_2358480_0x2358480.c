// Function: sub_2358480
// Address: 0x2358480
//
void __fastcall sub_2358480(unsigned __int64 *a1, int *a2)
{
  int v3; // edx
  char v4; // al
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // [rsp+8h] [rbp-58h] BYREF
  int v12; // [rsp+10h] [rbp-50h]
  char v13; // [rsp+14h] [rbp-4Ch]
  __int64 v14[2]; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v15[2]; // [rsp+28h] [rbp-38h] BYREF
  char v16; // [rsp+38h] [rbp-28h]

  v3 = *a2;
  v4 = *((_BYTE *)a2 + 4);
  v14[0] = (__int64)v15;
  v5 = (_BYTE *)*((_QWORD *)a2 + 1);
  v12 = v3;
  v6 = *((_QWORD *)a2 + 2);
  v13 = v4;
  sub_2303810(v14, v5, (__int64)&v5[v6]);
  v16 = *((_BYTE *)a2 + 40);
  v7 = sub_22077B0(0x38u);
  v8 = v7;
  if ( v7 )
  {
    v9 = (_BYTE *)v14[0];
    v10 = v14[1];
    *(_QWORD *)v7 = &unk_4A0D578;
    *(_DWORD *)(v7 + 8) = v12;
    *(_BYTE *)(v7 + 12) = v13;
    *(_QWORD *)(v7 + 16) = v7 + 32;
    sub_2303810((__int64 *)(v7 + 16), v9, (__int64)&v9[v10]);
    *(_BYTE *)(v8 + 48) = v16;
  }
  v11 = v8;
  sub_2356EF0(a1, &v11);
  if ( v11 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v11 + 8LL))(v11);
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
}
