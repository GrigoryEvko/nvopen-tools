// Function: sub_B24A20
// Address: 0xb24a20
//
void __fastcall sub_B24A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  unsigned int v7; // eax
  unsigned int v8; // edx
  __int64 v9; // rcx
  __int64 *v10; // r10
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 *v13; // rcx
  __int64 v14; // rsi
  __int64 *v15; // rbx
  __int64 *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 *v20; // rcx
  __int64 *v21; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+8h] [rbp-B8h]
  _BYTE v23[176]; // [rsp+10h] [rbp-B0h] BYREF

  if ( a3 )
  {
    v6 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    v7 = *(_DWORD *)(a3 + 44) + 1;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  v8 = *(_DWORD *)(a1 + 32);
  if ( v7 < v8 )
  {
    v9 = *(_QWORD *)(a1 + 24);
    v10 = *(__int64 **)(v9 + 8 * v6);
    if ( v10 )
    {
      *(_BYTE *)(a1 + 112) = 0;
      if ( a4 )
      {
        v11 = (unsigned int)(*(_DWORD *)(a4 + 44) + 1);
        v12 = *(_DWORD *)(a4 + 44) + 1;
      }
      else
      {
        v11 = 0;
        v12 = 0;
      }
      if ( v8 > v12 && (v13 = *(__int64 **)(v9 + 8 * v11)) != 0 )
      {
        sub_B242C0(a1, a2, *v10, v13);
      }
      else
      {
        v14 = a2;
        v21 = (__int64 *)v23;
        v22 = 0x800000000LL;
        sub_B1EAC0(a1, a2, a4, v10, (__int64)&v21);
        v15 = v21;
        v16 = &v21[2 * (unsigned int)v22];
        if ( v21 != v16 )
        {
          do
          {
            v17 = *v15;
            if ( *v15 )
            {
              v18 = (unsigned int)(*(_DWORD *)(v17 + 44) + 1);
              v19 = *(_DWORD *)(v17 + 44) + 1;
            }
            else
            {
              v18 = 0;
              v19 = 0;
            }
            if ( v19 >= *(_DWORD *)(a1 + 32) )
              BUG();
            v20 = (__int64 *)v15[1];
            v14 = a2;
            v15 += 2;
            sub_B242C0(a1, a2, **(_QWORD **)(*(_QWORD *)(a1 + 24) + 8 * v18), v20);
          }
          while ( v16 != v15 );
          v16 = v21;
        }
        if ( v16 != (__int64 *)v23 )
          _libc_free(v16, v14);
      }
    }
  }
}
