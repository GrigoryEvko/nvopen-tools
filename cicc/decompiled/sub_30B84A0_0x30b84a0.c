// Function: sub_30B84A0
// Address: 0x30b84a0
//
char __fastcall sub_30B84A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rax
  __int64 v6; // r9
  int v9; // eax
  int v10; // r12d
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v18; // [rsp-78h] [rbp-78h]
  const void *v19; // [rsp-70h] [rbp-70h]
  __int64 v20; // [rsp-68h] [rbp-68h]
  __int64 v21; // [rsp-68h] [rbp-68h]
  int v22; // [rsp-50h] [rbp-50h]
  __int64 v23; // [rsp-50h] [rbp-50h]
  __int64 v24; // [rsp-48h] [rbp-48h] BYREF
  __int64 v25[8]; // [rsp-40h] [rbp-40h] BYREF

  LODWORD(v5) = *(_DWORD *)(a4 + 8);
  if ( (_DWORD)v5 )
  {
    v6 = a2;
    if ( *(_WORD *)(a2 + 24) != 8 || *(_QWORD *)(a2 + 40) == 2 )
    {
      v9 = (_DWORD)v5 - 1;
      v22 = v9;
      if ( v9 >= 0 )
      {
        v10 = v9;
        v11 = 8LL * v9;
        v19 = (const void *)(a3 + 16);
        sub_310BF50(a1, a2, *(_QWORD *)(*(_QWORD *)a4 + v11), &v24, v25);
        while ( 1 )
        {
          v6 = v24;
          if ( v22 == v10 )
          {
            v20 = v24;
            LOBYTE(v5) = sub_D968A0(v25[0]);
            v6 = v20;
            if ( !(_BYTE)v5 )
            {
              *(_DWORD *)(a3 + 8) = 0;
              *(_DWORD *)(a4 + 8) = 0;
              return (char)v5;
            }
          }
          else
          {
            v12 = *(unsigned int *)(a3 + 8);
            a5 = v25[0];
            if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
            {
              v18 = v24;
              v21 = v25[0];
              sub_C8D5F0(a3, v19, v12 + 1, 8u, v25[0], v24);
              v12 = *(unsigned int *)(a3 + 8);
              v6 = v18;
              a5 = v21;
            }
            *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = a5;
            ++*(_DWORD *)(a3 + 8);
          }
          --v10;
          v11 -= 8;
          if ( v10 == -1 )
            break;
          sub_310BF50(a1, v6, *(_QWORD *)(*(_QWORD *)a4 + v11), &v24, v25);
        }
      }
      v13 = *(unsigned int *)(a3 + 8);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v23 = v6;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v13 + 1, 8u, a5, v6);
        v13 = *(unsigned int *)(a3 + 8);
        v6 = v23;
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = v6;
      v14 = *(__int64 **)a3;
      v15 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v15;
      v5 = &v14[v15];
      if ( v14 != v5 )
      {
        while ( v14 < --v5 )
        {
          v16 = *v14++;
          *(v14 - 1) = *v5;
          *v5 = v16;
        }
      }
    }
  }
  return (char)v5;
}
