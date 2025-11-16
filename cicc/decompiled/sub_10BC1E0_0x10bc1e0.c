// Function: sub_10BC1E0
// Address: 0x10bc1e0
//
__int64 __fastcall sub_10BC1E0(unsigned __int8 *a1, __int64 *a2)
{
  unsigned __int8 *v2; // rdx
  unsigned __int8 *v3; // rbx
  int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v9; // r11
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // ebx
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+8h] [rbp-98h]
  int v24[8]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v25; // [rsp+30h] [rbp-70h]
  _BYTE v26[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v27; // [rsp+60h] [rbp-40h]

  v2 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
  v3 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
  v4 = *a1;
  v5 = *((_QWORD *)v2 + 2);
  v6 = *((_QWORD *)v3 + 2);
  if ( !v5 || *(_QWORD *)(v5 + 8) || v4 != *v2 || (v8 = *((_QWORD *)v2 - 8)) == 0 || (v9 = *((_QWORD *)v2 - 4)) == 0 )
  {
    if ( v6 )
    {
      if ( !*(_QWORD *)(v6 + 8) && v4 == *v3 )
      {
        v8 = *((_QWORD *)v3 - 8);
        if ( v8 )
        {
          v9 = *((_QWORD *)v3 - 4);
          if ( v9 )
          {
            if ( v5 )
            {
              v3 = (unsigned __int8 *)*((_QWORD *)a1 - 8);
              if ( !*(_QWORD *)(v5 + 8) )
                goto LABEL_18;
            }
          }
        }
      }
    }
    return 0;
  }
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    return 0;
LABEL_18:
  result = 0;
  if ( *(_BYTE *)v8 > 0x15u && *(_BYTE *)v9 > 0x15u && *v3 > 0x15u )
  {
    v10 = *(_QWORD *)(v8 + 16);
    v11 = v4 - 29;
    if ( !v10 || *(_QWORD *)(v10 + 8) )
    {
      v12 = a2[10];
      v22 = v9;
      v25 = 257;
      v13 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, unsigned __int8 *))(*(_QWORD *)v12 + 16LL))(
              v12,
              v11,
              v9,
              v3);
      if ( !v13 )
      {
        v27 = 257;
        v13 = sub_B504D0(v11, v22, (__int64)v3, (__int64)v26, 0, 0);
        if ( (unsigned __int8)sub_920620(v13) )
        {
          v16 = a2[12];
          v17 = *((_DWORD *)a2 + 26);
          if ( v16 )
            sub_B99FD0(v13, 3u, v16);
          sub_B45150(v13, v17);
        }
        (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
          a2[11],
          v13,
          v24,
          a2[7],
          a2[8]);
        v18 = *a2 + 16LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v18 )
        {
          v19 = *a2;
          do
          {
            v20 = *(_QWORD *)(v19 + 8);
            v21 = *(_DWORD *)v19;
            v19 += 16;
            sub_B99FD0(v13, v21, v20);
          }
          while ( v18 != v19 );
        }
      }
      v27 = 257;
      return sub_B504D0(v11, v13, v8, (__int64)v26, 0, 0);
    }
    else
    {
      v14 = *(_QWORD *)(v9 + 16);
      if ( !v14 || (result = *(_QWORD *)(v14 + 8)) != 0 )
      {
        v23 = v9;
        v27 = 257;
        v15 = sub_10BBE20(a2, v11, v8, (__int64)v3, v24[0], 0, (__int64)v26, 0);
        v27 = 257;
        return sub_B504D0(v11, v15, v23, (__int64)v26, 0, 0);
      }
    }
  }
  return result;
}
