// Function: sub_2ECE620
// Address: 0x2ece620
//
__int64 __fastcall sub_2ECE620(__int64 a1, unsigned int a2, int a3, unsigned int a4)
{
  __int64 v6; // rdi
  unsigned int v8; // r12d
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned int v15; // r9d
  unsigned int *v16; // rax
  unsigned int v17; // [rsp+Ch] [rbp-44h] BYREF
  unsigned int *v18[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 (__fastcall *v19)(unsigned int **, unsigned int **, int); // [rsp+20h] [rbp-30h]
  __int64 (__fastcall *v20)(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD), unsigned int *, unsigned int *, unsigned int *); // [rsp+28h] [rbp-28h]

  v6 = *(_QWORD *)(a1 + 8);
  v17 = a2;
  if ( !v6 || !(unsigned __int8)sub_2FF85F0() )
  {
    v8 = *(_DWORD *)(*(_QWORD *)(a1 + 336) + 4LL * v17);
    if ( v8 != -1 )
    {
      if ( *(_DWORD *)(a1 + 24) != 1 )
      {
        v8 += a3;
        if ( *(_DWORD *)(a1 + 164) >= v8 )
          return *(unsigned int *)(a1 + 164);
      }
      return v8;
    }
    return *(unsigned int *)(a1 + 164);
  }
  v10 = *(_QWORD *)(a1 + 304);
  v11 = a1 + 296;
  if ( *(_DWORD *)(a1 + 24) == 1 )
  {
    if ( v10 )
    {
      v12 = a1 + 296;
      do
      {
        if ( *(_DWORD *)(v10 + 32) < v17 )
        {
          v10 = *(_QWORD *)(v10 + 24);
        }
        else
        {
          v12 = v10;
          v10 = *(_QWORD *)(v10 + 16);
        }
      }
      while ( v10 );
      if ( v11 != v12 && v17 >= *(_DWORD *)(v12 + 32) )
        goto LABEL_30;
    }
    else
    {
      v12 = a1 + 296;
    }
    v18[0] = &v17;
    v12 = sub_2ECE550((_QWORD *)(a1 + 288), v12, v18);
LABEL_30:
    v15 = *(_DWORD *)(a1 + 164);
    v16 = (unsigned int *)byte_2EC0AF0;
    goto LABEL_18;
  }
  if ( v10 )
  {
    v12 = a1 + 296;
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(v10 + 16);
        v14 = *(_QWORD *)(v10 + 24);
        if ( *(_DWORD *)(v10 + 32) >= v17 )
          break;
        v10 = *(_QWORD *)(v10 + 24);
        if ( !v14 )
          goto LABEL_14;
      }
      v12 = v10;
      v10 = *(_QWORD *)(v10 + 16);
    }
    while ( v13 );
LABEL_14:
    if ( v11 != v12 && v17 >= *(_DWORD *)(v12 + 32) )
      goto LABEL_17;
  }
  else
  {
    v12 = a1 + 296;
  }
  v18[0] = &v17;
  v12 = sub_2ECE550((_QWORD *)(a1 + 288), v12, v18);
LABEL_17:
  v15 = *(_DWORD *)(a1 + 164);
  v16 = (unsigned int *)sub_2EC0AD0;
LABEL_18:
  v18[0] = v16;
  v20 = sub_2EC0BF0;
  v19 = (__int64 (__fastcall *)(unsigned int **, unsigned int **, int))sub_2EC0C10;
  v8 = sub_2ECA350((_QWORD *)(v12 + 40), v15, a4, a3, (__int64)v18);
  if ( !v19 )
    return v8;
  v19(v18, v18, 3);
  return v8;
}
