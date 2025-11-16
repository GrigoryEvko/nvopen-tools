// Function: sub_2F40A90
// Address: 0x2f40a90
//
__int64 __fastcall sub_2F40A90(__int64 a1, __int64 a2, unsigned __int16 **a3, unsigned __int8 a4, __int64 a5)
{
  unsigned __int16 **v5; // r13
  __int64 result; // rax
  __int64 v8; // r9
  int v9; // ecx
  int v10; // r15d
  unsigned __int16 **v11; // rax
  int v12; // r13d
  unsigned __int16 **v13; // r15
  unsigned int v14; // r14d
  int v15; // r11d
  unsigned __int16 *v16; // r10
  __int64 v17; // r9
  unsigned __int16 *v18; // rsi
  __int64 v19; // r9
  int v20; // r11d
  int v21; // ebx
  unsigned __int16 **v22; // r14
  unsigned __int16 *v23; // r13
  int *v24; // r11
  unsigned __int16 *v25; // rsi
  __int64 v26; // r9
  int v27; // r10d
  unsigned int v30; // [rsp+20h] [rbp-60h]
  unsigned int v31; // [rsp+24h] [rbp-5Ch]
  int v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+28h] [rbp-58h]
  int v34; // [rsp+3Ch] [rbp-44h] BYREF
  __int64 v35; // [rsp+40h] [rbp-40h] BYREF
  __int64 v36; // [rsp+48h] [rbp-38h]

  v5 = a3;
  v35 = 0xFFFFFFFFLL;
  v31 = a4;
  v36 = sub_2F50FE0(a1, a2, a3, a4);
  result = 0;
  if ( BYTE4(v36) )
  {
    v8 = (int)v36;
    if ( a4 != 0xFF )
    {
      LODWORD(v35) = 0;
      HIDWORD(v35) = *(_DWORD *)(a2 + 116);
    }
    v9 = *((_DWORD *)v5 + 18);
    v32 = v9;
    v10 = -*((_DWORD *)v5 + 2);
    if ( (_DWORD)v36 )
    {
      v21 = v36 - 1;
      if ( (int)v36 - 1 > v9 )
      {
        v21 = *((_DWORD *)v5 + 18);
      }
      else if ( v21 < v9 )
      {
        if ( (int)v36 >= v9 || (int)v36 < 0 )
        {
          v21 = v36;
        }
        else
        {
          v33 = (__int64)v5[1];
          v22 = v5;
          v23 = v5[7];
          v24 = &v34;
          do
          {
            v21 = v8;
            if ( (unsigned int)v23[v8] - 1 > 0x3FFFFFFE )
              break;
            v34 = v23[v8];
            v25 = &(*v22)[v33];
            if ( v25 == sub_2F3FB20(*v22, (__int64)v25, v24) )
              break;
            v8 = v26 + 1;
            ++v21;
          }
          while ( v27 > (int)v8 );
          v5 = v22;
        }
      }
      v32 = v21;
      v30 = 0;
      if ( v10 != v21 )
      {
LABEL_6:
        v11 = v5;
        v12 = v10;
        v13 = v11;
        do
        {
LABEL_8:
          if ( v12 < 0 )
            v14 = (*v13)[(_QWORD)v13[1] + v12];
          else
            v14 = v13[7][v12];
          if ( (unsigned __int8)sub_2F510F0(a1, v31, v14) )
          {
            if ( (unsigned __int8)sub_2F405D0(a1, a2, v14, 0, (float *)&v35, a5) )
            {
              v30 = v14;
              if ( v12 < 0 )
                break;
            }
          }
          v15 = *((_DWORD *)v13 + 18);
          if ( v15 > v12 && v15 > ++v12 && v12 >= 0 )
          {
            v16 = v13[7];
            v17 = v12;
            while ( 1 )
            {
              v12 = v17;
              if ( (unsigned int)v16[v17] - 1 > 0x3FFFFFFE )
                break;
              v34 = v16[v17];
              v18 = &(*v13)[(_QWORD)v13[1]];
              if ( v18 == sub_2F3FB20(*v13, (__int64)v18, &v34) )
                break;
              v17 = v19 + 1;
              ++v12;
              if ( v20 <= (int)v17 )
              {
                if ( v32 != v12 )
                  goto LABEL_8;
                return v30;
              }
            }
          }
        }
        while ( v32 != v12 );
      }
    }
    else
    {
      v30 = 0;
      if ( v10 != v9 )
        goto LABEL_6;
    }
    return v30;
  }
  return result;
}
