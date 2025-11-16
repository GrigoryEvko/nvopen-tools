// Function: sub_2E6F920
// Address: 0x2e6f920
//
__int64 __fastcall sub_2E6F920(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 result; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rbx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // [rsp+0h] [rbp-40h]
  __int64 v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v8 = *(_DWORD *)(a2 + 24) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  if ( v8 >= *(_DWORD *)(a3 + 32) || (result = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v7)) == 0 )
  {
    v13 = *(_QWORD *)(sub_2E6F1C0(a1, a2, v7, a4, a5, a6) + 16);
    if ( v13 )
    {
      v14 = (unsigned int)(*(_DWORD *)(v13 + 24) + 1);
      v15 = *(_DWORD *)(v13 + 24) + 1;
    }
    else
    {
      v14 = 0;
      v15 = 0;
    }
    if ( v15 >= *(_DWORD *)(a3 + 32) || (v14 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v14)) == 0 )
    {
      v17 = *(_QWORD *)(sub_2E6F1C0(a1, v13, v14, v10, v11, v12) + 16);
      if ( v17 )
      {
        v20 = (unsigned int)(*(_DWORD *)(v17 + 24) + 1);
        v21 = *(_DWORD *)(v17 + 24) + 1;
      }
      else
      {
        v20 = 0;
        v21 = 0;
      }
      if ( v21 < *(_DWORD *)(a3 + 32) && (v20 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v20)) != 0 )
      {
        v14 = sub_2E6E2F0(a3, v13, v20);
      }
      else
      {
        v23 = *(_QWORD *)(sub_2E6F1C0(a1, v17, v20, v16, v18, v19) + 16);
        if ( v23 )
        {
          v25 = (unsigned int)(*(_DWORD *)(v23 + 24) + 1);
          v26 = *(_DWORD *)(v23 + 24) + 1;
        }
        else
        {
          v25 = 0;
          v26 = 0;
        }
        if ( v26 >= *(_DWORD *)(a3 + 32) || (v25 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v25)) == 0 )
        {
          v38 = v23;
          v28 = sub_2E6F1C0(a1, v23, v25, v22, v23, v24);
          v30 = v38;
          v31 = *(_QWORD *)(v28 + 16);
          if ( v31 )
          {
            v32 = (unsigned int)(*(_DWORD *)(v31 + 24) + 1);
            v33 = *(_DWORD *)(v31 + 24) + 1;
          }
          else
          {
            v32 = 0;
            v33 = 0;
          }
          if ( v33 >= *(_DWORD *)(a3 + 32) || (v32 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v32)) == 0 )
          {
            v37 = v38;
            v39 = v31;
            v34 = sub_2E6F1C0(a1, v31, v32, v29, v30, v31);
            v35 = sub_2E6F920(a1, *(_QWORD *)(v34 + 16), a3);
            v36 = sub_2E6E2F0(a3, v39, v35);
            v30 = v37;
            v32 = v36;
          }
          v25 = sub_2E6E2F0(a3, v30, v32);
        }
        v27 = sub_2E6E2F0(a3, v17, v25);
        v14 = sub_2E6E2F0(a3, v13, v27);
      }
    }
    return sub_2E6E2F0(a3, a2, v14);
  }
  return result;
}
