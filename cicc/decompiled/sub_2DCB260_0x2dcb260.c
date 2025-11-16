// Function: sub_2DCB260
// Address: 0x2dcb260
//
__int64 __fastcall sub_2DCB260(unsigned int *a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned int *v6; // r12
  unsigned int *v7; // r14
  unsigned int v8; // ebx
  unsigned int v9; // ebx
  unsigned int v10; // eax
  unsigned int *v11; // r14
  unsigned int *v12; // r12
  unsigned int v13; // ebx
  unsigned int v14; // ebx
  unsigned int v15; // eax
  unsigned int v16; // ebx
  unsigned int v17; // eax
  unsigned int v18; // edx
  __int64 v19; // rbx
  __int64 v20; // r12
  unsigned int v21; // ecx
  unsigned int v22; // ebx
  bool v23; // cc
  unsigned int v24; // ebx
  __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned int *v26; // [rsp+10h] [rbp-40h]
  unsigned int *v27; // [rsp+18h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v26 = a2;
  v25 = a3;
  if ( (char *)a2 - (char *)a1 <= 64 )
    return result;
  if ( !a3 )
  {
    v27 = a2;
    goto LABEL_19;
  }
  while ( 2 )
  {
    v6 = &a1[result >> 3];
    --v25;
    v7 = v26 - 1;
    v8 = sub_2DCA6E0(*(_QWORD *)(a4 + 8), a1[1]);
    if ( v8 <= (unsigned int)sub_2DCA6E0(*(_QWORD *)(a4 + 8), *v6) )
    {
      v16 = sub_2DCA6E0(*(_QWORD *)(a4 + 8), a1[1]);
      if ( v16 <= (unsigned int)sub_2DCA6E0(*(_QWORD *)(a4 + 8), *(v26 - 1)) )
      {
        v24 = sub_2DCA6E0(*(_QWORD *)(a4 + 8), *v6);
        v23 = v24 <= (unsigned int)sub_2DCA6E0(*(_QWORD *)(a4 + 8), *v7);
        v17 = *a1;
        if ( v23 )
        {
          *a1 = *v6;
          *v6 = v17;
          goto LABEL_6;
        }
        goto LABEL_24;
      }
      v17 = *a1;
LABEL_17:
      v18 = a1[1];
      a1[1] = v17;
      *a1 = v18;
      goto LABEL_6;
    }
    v9 = sub_2DCA6E0(*(_QWORD *)(a4 + 8), *v6);
    if ( v9 <= (unsigned int)sub_2DCA6E0(*(_QWORD *)(a4 + 8), *(v26 - 1)) )
    {
      v22 = sub_2DCA6E0(*(_QWORD *)(a4 + 8), a1[1]);
      v23 = v22 <= (unsigned int)sub_2DCA6E0(*(_QWORD *)(a4 + 8), *v7);
      v17 = *a1;
      if ( !v23 )
      {
LABEL_24:
        *a1 = *(v26 - 1);
        *(v26 - 1) = v17;
        goto LABEL_6;
      }
      goto LABEL_17;
    }
    v10 = *a1;
    *a1 = *v6;
    *v6 = v10;
LABEL_6:
    v11 = a1 + 1;
    v12 = v26;
    while ( 1 )
    {
      v27 = v11;
      v13 = sub_2DCA6E0(*(_QWORD *)(a4 + 8), *v11);
      if ( v13 > (unsigned int)sub_2DCA6E0(*(_QWORD *)(a4 + 8), *a1) )
        goto LABEL_7;
      do
      {
        --v12;
        v14 = sub_2DCA6E0(*(_QWORD *)(a4 + 8), *a1);
      }
      while ( v14 > (unsigned int)sub_2DCA6E0(*(_QWORD *)(a4 + 8), *v12) );
      if ( v11 >= v12 )
        break;
      v15 = *v11;
      *v11 = *v12;
      *v12 = v15;
LABEL_7:
      ++v11;
    }
    sub_2DCB260(v11, v26, v25, a4);
    result = (char *)v11 - (char *)a1;
    if ( (char *)v11 - (char *)a1 > 64 )
    {
      if ( v25 )
      {
        v26 = v11;
        continue;
      }
LABEL_19:
      v19 = result >> 2;
      v20 = ((result >> 2) - 2) >> 1;
      sub_2DCA840((__int64)a1, v20, result >> 2, a1[v20], a4);
      do
      {
        --v20;
        sub_2DCA840((__int64)a1, v20, v19, a1[v20], a4);
      }
      while ( v20 );
      do
      {
        v21 = *--v27;
        *v27 = *a1;
        result = sub_2DCA840((__int64)a1, 0, v27 - a1, v21, a4);
      }
      while ( (char *)v27 - (char *)a1 > 4 );
    }
    return result;
  }
}
