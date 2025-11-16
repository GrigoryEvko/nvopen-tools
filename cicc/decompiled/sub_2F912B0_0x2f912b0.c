// Function: sub_2F912B0
// Address: 0x2f912b0
//
__int64 __fastcall sub_2F912B0(__int64 a1, __int64 a2, __int64 *a3)
{
  _DWORD *v5; // rdx
  _DWORD *v6; // rdx
  _QWORD v7[3]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v8; // [rsp+18h] [rbp-58h]
  _DWORD *v9; // [rsp+20h] [rbp-50h]
  __int64 v10; // [rsp+28h] [rbp-48h]
  __int64 v11; // [rsp+30h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v10 = 0x100000000LL;
  v11 = a1;
  v7[0] = &unk_49DD210;
  v7[1] = 0;
  v7[2] = 0;
  v8 = 0;
  v9 = 0;
  sub_CB5980((__int64)v7, 0, 0, 0);
  if ( a3 == (__int64 *)(a2 + 72) )
  {
    v6 = v9;
    if ( (unsigned __int64)(v8 - (_QWORD)v9) <= 6 )
    {
      sub_CB6200((__int64)v7, "<entry>", 7u);
    }
    else
    {
      *v9 = 1953391932;
      *((_WORD *)v6 + 2) = 31090;
      *((_BYTE *)v6 + 6) = 62;
      v9 = (_DWORD *)((char *)v9 + 7);
    }
  }
  else if ( a3 == (__int64 *)(a2 + 328) )
  {
    v5 = v9;
    if ( (unsigned __int64)(v8 - (_QWORD)v9) <= 5 )
    {
      sub_CB6200((__int64)v7, "<exit>", 6u);
    }
    else
    {
      *v9 = 1769497916;
      *((_WORD *)v5 + 2) = 15988;
      v9 = (_DWORD *)((char *)v9 + 6);
    }
  }
  else
  {
    sub_2E91850(*a3, (__int64)v7, 1u, 0, 0, 1, 0);
  }
  v7[0] = &unk_49DD210;
  sub_CB5840((__int64)v7);
  return a1;
}
