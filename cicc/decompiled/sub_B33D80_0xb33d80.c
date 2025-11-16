// Function: sub_B33D80
// Address: 0xb33d80
//
__int64 __fastcall sub_B33D80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // rax
  __int64 v9; // r13
  bool v10; // al
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r12
  unsigned int *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // [rsp+Ch] [rbp-94h]
  _QWORD v19[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v20; // [rsp+30h] [rbp-70h]
  _DWORD v21[8]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v22; // [rsp+60h] [rbp-40h]

  v4 = a2;
  v6 = *(_DWORD *)(a2 + 32);
  if ( v6 <= 0x40 )
    v7 = *(_QWORD *)(a2 + 24) == 0;
  else
    v7 = v6 == (unsigned int)sub_C444A0(a2 + 24);
  if ( !v7 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    v21[1] = 0;
    v19[0] = v8;
    v9 = sub_B33D10(a1, 0x1EDu, (__int64)v19, 1, 0, 0, v21[0], a3);
    if ( *(_DWORD *)(a2 + 32) <= 0x40u )
    {
      v10 = *(_QWORD *)(a2 + 24) == 1;
    }
    else
    {
      v18 = *(_DWORD *)(a2 + 32);
      v10 = v18 - 1 == (unsigned int)sub_C444A0(a2 + 24);
    }
    if ( !v10 )
    {
      v12 = *(_QWORD *)(a1 + 80);
      v20 = 257;
      v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v12 + 32LL))(
              v12,
              17,
              v9,
              a2,
              0,
              0);
      if ( v13 )
      {
        return v13;
      }
      else
      {
        v22 = 257;
        v9 = sub_B504D0(17, v9, a2, v21, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
          *(_QWORD *)(a1 + 88),
          v9,
          v19,
          *(_QWORD *)(a1 + 56),
          *(_QWORD *)(a1 + 64));
        v14 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
        if ( *(_QWORD *)a1 != v14 )
        {
          v15 = *(unsigned int **)a1;
          do
          {
            v16 = *((_QWORD *)v15 + 1);
            v17 = *v15;
            v15 += 4;
            sub_B99FD0(v9, v17, v16);
          }
          while ( (unsigned int *)v14 != v15 );
        }
      }
    }
    return v9;
  }
  return v4;
}
