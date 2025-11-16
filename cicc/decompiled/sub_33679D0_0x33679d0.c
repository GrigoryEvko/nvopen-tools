// Function: sub_33679D0
// Address: 0x33679d0
//
__int64 __fastcall sub_33679D0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rbx
  unsigned int v5; // r12d
  unsigned int v6; // edx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+18h] [rbp-68h]
  unsigned int v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+18h] [rbp-68h]
  __int64 v22; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v23; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+38h] [rbp-48h]
  char v25; // [rsp+40h] [rbp-40h]

  result = a2 + 24 * a3;
  v17 = result;
  if ( a2 != result )
  {
    v4 = a2;
    v5 = 0;
    do
    {
      while ( 1 )
      {
        v18 = sub_CA1930((_BYTE *)(v4 + 8));
        sub_AF47B0((__int64)&v23, *(unsigned __int64 **)(**a1 + 16), *(unsigned __int64 **)(**a1 + 24));
        result = v18;
        if ( v25 )
        {
          if ( v5 >= (unsigned __int64)v23 )
            return result;
          v6 = (_DWORD)v23 - v5;
          if ( v5 + (unsigned int)v18 <= (unsigned __int64)v23 )
            v6 = v18;
        }
        else
        {
          v6 = v18;
        }
        v23 = (_QWORD *)sub_B0E470(**a1, v5, v6);
        v24 = v7;
        v5 += sub_CA1930((_BYTE *)(v4 + 8));
        if ( !(_BYTE)v24 )
          break;
        sub_3367790(a1[5], *(_DWORD *)v4, v23, *(_DWORD *)a1[6] != 0);
        v9 = v8;
        result = a1[1][120];
        v10 = *(unsigned int *)(result + 320);
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(result + 324) )
        {
          v16 = v9;
          v21 = a1[1][120];
          sub_C8D5F0(result + 312, (const void *)(result + 328), v10 + 1, 8u, v9, v10 + 1);
          result = v21;
          v9 = v16;
          v10 = *(unsigned int *)(v21 + 320);
        }
        v4 += 24;
        *(_QWORD *)(*(_QWORD *)(result + 312) + 8 * v10) = v9;
        ++*(_DWORD *)(result + 320);
        if ( v17 == v4 )
          return result;
      }
      v11 = a1[1];
      v15 = v11[108];
      v19 = *((_DWORD *)v11 + 212);
      sub_B10CB0(&v22, *a1[4]);
      v12 = sub_ACADE0(*(__int64 ***)(*a1[3] + 8));
      v13 = sub_33E5DB0(v15, *a1[2], **a1, v12, &v22, v19);
      v14 = v13;
      if ( v22 )
      {
        v20 = v13;
        sub_B91220((__int64)&v22, v22);
        v14 = v20;
      }
      v4 += 24;
      result = sub_33F99B0(a1[1][108], v14, 0);
    }
    while ( v17 != v4 );
  }
  return result;
}
