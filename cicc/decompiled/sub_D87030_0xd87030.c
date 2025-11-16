// Function: sub_D87030
// Address: 0xd87030
//
__int64 __fastcall sub_D87030(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 *v5; // rdi
  unsigned __int64 v6; // rbx
  __int64 v7; // r13
  int v8; // eax
  unsigned int v9; // eax
  unsigned __int64 v10; // r12
  unsigned int i; // ecx
  __int64 v12; // rsi
  __int64 v13; // rdi
  bool v14; // cc
  int v15; // ecx
  __int64 v16; // rdi
  int v17; // ecx
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // [rsp+8h] [rbp-88h]
  __int64 v21; // [rsp+10h] [rbp-80h]
  unsigned int v22; // [rsp+18h] [rbp-78h]
  int v23; // [rsp+1Ch] [rbp-74h]
  __int64 v25; // [rsp+28h] [rbp-68h]
  __int64 v26; // [rsp+38h] [rbp-58h]

  if ( a1 != a2 )
  {
    result = a1 + 48;
    if ( a2 != a1 + 48 )
    {
      v4 = a1 + 96;
      do
      {
        while ( 1 )
        {
          v25 = v4;
          v6 = *(_QWORD *)(v4 - 48);
          v7 = v4 - 48;
          if ( v6 < *(_QWORD *)a1
            || v6 == *(_QWORD *)a1
            && *(_QWORD *)(*(_QWORD *)(v4 - 40) & 0xFFFFFFFFFFFFFFF8LL) < *(_QWORD *)(*(_QWORD *)(a1 + 8)
                                                                                    & 0xFFFFFFFFFFFFFFF8LL) )
          {
            break;
          }
          v5 = (__int64 *)(v4 - 48);
          v4 += 48;
          result = sub_D86EE0(v5);
          if ( a2 == v25 )
            return result;
        }
        v26 = *(_QWORD *)(v4 - 40);
        v8 = *(_DWORD *)(v4 - 24);
        *(_DWORD *)(v4 - 24) = 0;
        v23 = v8;
        v21 = *(_QWORD *)(v4 - 32);
        v9 = *(_DWORD *)(v4 - 8);
        *(_DWORD *)(v4 - 8) = 0;
        v22 = v9;
        v20 = *(_QWORD *)(v4 - 16);
        v10 = 0xAAAAAAAAAAAAAAABLL * ((v7 - a1) >> 4);
        if ( v7 - a1 > 0 )
        {
          for ( i = 0; ; i = *(_DWORD *)(v7 + 24) )
          {
            v12 = *(_QWORD *)(v7 - 48);
            v7 -= 48;
            *(_QWORD *)(v7 + 48) = v12;
            *(_QWORD *)(v7 + 56) = *(_QWORD *)(v7 + 8);
            if ( i > 0x40 )
            {
              v13 = *(_QWORD *)(v7 + 64);
              if ( v13 )
                j_j___libc_free_0_0(v13);
            }
            v14 = *(_DWORD *)(v7 + 88) <= 0x40u;
            *(_QWORD *)(v7 + 64) = *(_QWORD *)(v7 + 16);
            v15 = *(_DWORD *)(v7 + 24);
            *(_DWORD *)(v7 + 24) = 0;
            *(_DWORD *)(v7 + 72) = v15;
            if ( !v14 )
            {
              v16 = *(_QWORD *)(v7 + 80);
              if ( v16 )
                j_j___libc_free_0_0(v16);
            }
            *(_QWORD *)(v7 + 80) = *(_QWORD *)(v7 + 32);
            v17 = *(_DWORD *)(v7 + 40);
            *(_DWORD *)(v7 + 40) = 0;
            *(_DWORD *)(v7 + 88) = v17;
            if ( !--v10 )
              break;
          }
        }
        v14 = *(_DWORD *)(a1 + 24) <= 0x40u;
        *(_QWORD *)a1 = v6;
        *(_QWORD *)(a1 + 8) = v26;
        if ( !v14 )
        {
          v18 = *(_QWORD *)(a1 + 16);
          if ( v18 )
            j_j___libc_free_0_0(v18);
        }
        v14 = *(_DWORD *)(a1 + 40) <= 0x40u;
        *(_QWORD *)(a1 + 16) = v21;
        *(_DWORD *)(a1 + 24) = v23;
        if ( !v14 )
        {
          v19 = *(_QWORD *)(a1 + 32);
          if ( v19 )
            j_j___libc_free_0_0(v19);
        }
        v4 += 48;
        *(_QWORD *)(a1 + 32) = v20;
        result = v22;
        *(_DWORD *)(a1 + 40) = v22;
      }
      while ( a2 != v25 );
    }
  }
  return result;
}
