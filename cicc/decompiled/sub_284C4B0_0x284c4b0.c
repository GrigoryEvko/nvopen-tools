// Function: sub_284C4B0
// Address: 0x284c4b0
//
__int64 __fastcall sub_284C4B0(__int64 a1, __int64 *a2)
{
  bool v3; // zf
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // r13
  __int64 v7; // rsi
  __int64 *v8; // rbx
  __int64 v10; // rax
  bool v11; // cf
  __int64 v12; // rax
  __int64 *v13; // rax
  unsigned int v14; // [rsp+4h] [rbp-2Ch] BYREF
  unsigned __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_BYTE *)(a1 + 28) == 0;
  v15[0] = 0;
  v4 = *(__int64 **)(a1 + 8);
  if ( v3 )
    v5 = *(unsigned int *)(a1 + 16);
  else
    v5 = *(unsigned int *)(a1 + 20);
  v6 = &v4[v5];
  if ( v4 != v6 )
  {
    while ( 1 )
    {
      v7 = *v4;
      v8 = v4;
      if ( (unsigned __int64)*v4 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v6 == ++v4 )
        goto LABEL_6;
    }
    while ( v6 != v8 )
    {
      v10 = sub_FDD860(a2, v7);
      v11 = __CFADD__(v15[0], v10);
      v12 = v15[0] + v10;
      v15[0] = v11 ? -1LL : v12;
      v13 = v8 + 1;
      if ( v8 + 1 == v6 )
        break;
      while ( 1 )
      {
        v7 = *v13;
        v8 = v13;
        if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v6 == ++v13 )
        {
          if ( (unsigned int)(*(_DWORD *)(a1 + 20) - *(_DWORD *)(a1 + 24)) <= 1 )
            return v15[0];
          goto LABEL_15;
        }
      }
    }
  }
LABEL_6:
  if ( (unsigned int)(*(_DWORD *)(a1 + 20) - *(_DWORD *)(a1 + 24)) > 1 )
  {
LABEL_15:
    sub_F02DB0(&v14, qword_5000888, 0x64u);
    sub_1098D40(v15, v14);
  }
  return v15[0];
}
