// Function: sub_B21280
// Address: 0xb21280
//
__int64 __fastcall sub_B21280(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+8h] [rbp-38h]
  __int64 v31; // [rsp+8h] [rbp-38h]

  if ( a2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v5 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  if ( v5 >= *(_DWORD *)(a3 + 56) || (result = *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8 * v4)) == 0 )
  {
    v7 = *(_QWORD *)(sub_B20CA0(a1, a2) + 16);
    if ( v7 )
    {
      v8 = (unsigned int)(*(_DWORD *)(v7 + 44) + 1);
      v9 = *(_DWORD *)(v7 + 44) + 1;
    }
    else
    {
      v8 = 0;
      v9 = 0;
    }
    if ( v9 >= *(_DWORD *)(a3 + 56) || (v10 = *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8 * v8)) == 0 )
    {
      v11 = *(_QWORD *)(sub_B20CA0(a1, v7) + 16);
      if ( v11 )
      {
        v12 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
        v13 = *(_DWORD *)(v11 + 44) + 1;
      }
      else
      {
        v12 = 0;
        v13 = 0;
      }
      if ( v13 < *(_DWORD *)(a3 + 56) && (v14 = *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8 * v12)) != 0 )
      {
        v10 = sub_B1BBB0(a3, v7, v14);
      }
      else
      {
        v15 = *(_QWORD *)(sub_B20CA0(a1, v11) + 16);
        if ( v15 )
        {
          v16 = (unsigned int)(*(_DWORD *)(v15 + 44) + 1);
          v17 = *(_DWORD *)(v15 + 44) + 1;
        }
        else
        {
          v16 = 0;
          v17 = 0;
        }
        if ( v17 >= *(_DWORD *)(a3 + 56) || (v18 = *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8 * v16)) == 0 )
        {
          v30 = v15;
          v20 = sub_B20CA0(a1, v15);
          v21 = v30;
          v22 = *(_QWORD *)(v20 + 16);
          if ( v22 )
          {
            v23 = (unsigned int)(*(_DWORD *)(v22 + 44) + 1);
            v24 = *(_DWORD *)(v22 + 44) + 1;
          }
          else
          {
            v23 = 0;
            v24 = 0;
          }
          if ( v24 >= *(_DWORD *)(a3 + 56) || (v25 = *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8 * v23)) == 0 )
          {
            v29 = v30;
            v31 = v22;
            v26 = sub_B20CA0(a1, v22);
            v27 = sub_B21280(a1, *(_QWORD *)(v26 + 16), a3);
            v28 = sub_B1BBB0(a3, v31, v27);
            v21 = v29;
            v25 = v28;
          }
          v18 = sub_B1BBB0(a3, v21, v25);
        }
        v19 = sub_B1BBB0(a3, v11, v18);
        v10 = sub_B1BBB0(a3, v7, v19);
      }
    }
    return sub_B1BBB0(a3, a2, v10);
  }
  return result;
}
