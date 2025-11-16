// Function: sub_1D444E0
// Address: 0x1d444e0
//
void __fastcall sub_1D444E0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // ebx
  int v5; // r15d
  int v6; // edx
  int v7; // r8d
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  int v15; // ebx
  __int64 v16; // [rsp-78h] [rbp-78h] BYREF
  __int64 v17; // [rsp-70h] [rbp-70h] BYREF
  __int64 (__fastcall **v18)(); // [rsp-68h] [rbp-68h] BYREF
  __int64 v19; // [rsp-60h] [rbp-60h]
  __int64 v20; // [rsp-58h] [rbp-58h]
  __int64 *v21; // [rsp-50h] [rbp-50h]
  __int64 *v22; // [rsp-48h] [rbp-48h]

  if ( a2 != a3 )
  {
    v4 = 0;
    v5 = *(_DWORD *)(a2 + 60);
    if ( v5 )
    {
      do
      {
        while ( !(unsigned __int8)sub_1D18C40(a2, v4) )
        {
          if ( ++v4 == v5 )
            goto LABEL_7;
        }
        v6 = v4;
        v7 = v4++;
        sub_1D306C0(a1, a2, v6, a3, v7, 0, 0, 1);
      }
      while ( v4 != v5 );
    }
LABEL_7:
    v8 = *(_QWORD *)(a1 + 664);
    v9 = *(_QWORD *)(a2 + 48);
    v20 = a1;
    v17 = 0;
    v19 = v8;
    *(_QWORD *)(a1 + 664) = &v18;
    v21 = &v16;
    v10 = &v17;
    v16 = v9;
    v18 = off_49F99D8;
    v22 = &v17;
    if ( v9 )
    {
      do
      {
        v11 = *(_QWORD *)(v9 + 16);
        sub_1D2D480(a1, v11, (unsigned int)v10);
        v12 = v16;
        do
        {
          v13 = *(_QWORD *)(v12 + 32);
          v16 = v13;
          if ( *(_QWORD *)v12 )
          {
            **(_QWORD **)(v12 + 24) = v13;
            if ( v13 )
              *(_QWORD *)(v13 + 24) = *(_QWORD *)(v12 + 24);
          }
          *(_QWORD *)v12 = a3;
          if ( a3 )
          {
            v14 = *(_QWORD *)(a3 + 48);
            *(_QWORD *)(v12 + 32) = v14;
            if ( v14 )
              *(_QWORD *)(v14 + 24) = v12 + 32;
            *(_QWORD *)(v12 + 24) = a3 + 48;
            *(_QWORD *)(a3 + 48) = v12;
          }
          if ( ((*(_BYTE *)(a2 + 26) & 4) != 0) != ((*(_BYTE *)(a3 + 26) & 4) != 0) )
            sub_1D18440((_QWORD *)a1, v11);
          v12 = v16;
        }
        while ( v16 != v17 && v11 == *(_QWORD *)(v16 + 16) );
        sub_1D446C0(a1, v11);
        v9 = v16;
      }
      while ( v17 != v16 );
    }
    *(_DWORD *)(a3 + 64) = *(_DWORD *)(a2 + 64);
    if ( a2 == *(_QWORD *)(a1 + 176) )
    {
      v15 = *(_DWORD *)(a1 + 184);
      nullsub_686();
      *(_QWORD *)(a1 + 176) = a3;
      *(_DWORD *)(a1 + 184) = v15;
      sub_1D23870();
    }
    *(_QWORD *)(v20 + 664) = v19;
  }
}
