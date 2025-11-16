// Function: sub_15D4190
// Address: 0x15d4190
//
void __fastcall sub_15D4190(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 *v10; // r8
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 *v13; // rbx
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  unsigned __int64 v16; // rdi
  char v17; // al
  __int64 *v18; // [rsp+8h] [rbp-78h]
  unsigned int v19; // [rsp+8h] [rbp-78h]
  __int64 v20[4]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v21; // [rsp+30h] [rbp-50h]
  unsigned int v22; // [rsp+40h] [rbp-40h]

  v6 = sub_15CC510(a1, a3);
  if ( v6 )
  {
    v7 = (__int64 *)v6;
    v18 = (__int64 *)sub_15CC510(a1, a4);
    if ( v18 )
    {
      v8 = sub_15CC590(a1, a3, a4);
      v9 = sub_15CC510(a1, v8);
      v10 = v18;
      if ( v18 != (__int64 *)v9 )
      {
        *(_BYTE *)(a1 + 72) = 0;
        if ( v7 != (__int64 *)v18[1] || (v17 = sub_15CF2F0(a1, a2, v18), v10 = v18, v17) )
        {
          v11 = sub_15CC590(a1, *v7, *v10);
          v12 = sub_15CC510(a1, v11);
          v13 = *(__int64 **)(v12 + 8);
          if ( v13 )
          {
            v19 = *(_DWORD *)(v12 + 16);
            sub_15CDE90((__int64)v20, a2);
            sub_15D2C40((__int64)v20, v11, 0, v19, a1, 0);
            sub_15D2F60(v20, a1, v19);
            sub_15D2410(v20, a1, v13);
            if ( v22 )
            {
              v14 = v21;
              v15 = &v21[9 * v22];
              do
              {
                if ( *v14 != -16 && *v14 != -8 )
                {
                  v16 = v14[5];
                  if ( (_QWORD *)v16 != v14 + 7 )
                    _libc_free(v16);
                }
                v14 += 9;
              }
              while ( v15 != v14 );
            }
            j___libc_free_0(v21);
            if ( v20[0] )
              j_j___libc_free_0(v20[0], v20[2] - v20[0]);
          }
          else
          {
            sub_15D3360(a1, a2);
          }
        }
        else
        {
          sub_15D39A0(a1, a2, (__int64)v18);
        }
      }
    }
  }
}
