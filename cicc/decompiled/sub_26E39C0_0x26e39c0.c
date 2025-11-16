// Function: sub_26E39C0
// Address: 0x26e39c0
//
__int64 __fastcall sub_26E39C0(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // r12
  __int64 v5; // r15
  __int64 *v6; // r13
  __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // rsi
  __int64 v10; // r14
  __int64 result; // rax
  const __m128i *v12; // r13
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rsi
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h]
  _QWORD *v19; // [rsp+20h] [rbp-40h] BYREF
  const __m128i *v20[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a3 + 1;
  v5 = *(_QWORD *)(a2 + 96);
  v18 = a2 + 80;
  while ( v18 != v5 )
  {
    if ( *(char *)(v5 + 33) >= 0 )
    {
      v6 = *(__int64 **)(v5 + 64);
      if ( v6 )
      {
        v7 = a3[2];
        if ( v7 )
        {
LABEL_7:
          v8 = *(_DWORD *)(v5 + 32);
          v9 = (__int64)v3;
          while ( 1 )
          {
            while ( *(_DWORD *)(v7 + 32) < v8 )
            {
              v7 = *(_QWORD *)(v7 + 24);
LABEL_12:
              if ( !v7 )
              {
LABEL_13:
                if ( (_QWORD *)v9 != v3
                  && *(_DWORD *)(v9 + 32) <= v8
                  && (*(_DWORD *)(v9 + 32) != v8 || *(_DWORD *)(v5 + 36) >= *(_DWORD *)(v9 + 36)) )
                {
                  *(_QWORD *)(v9 + 48) = 23;
                  *(_QWORD *)(v9 + 40) = "unknown.indirect.callee";
                  goto LABEL_17;
                }
                goto LABEL_20;
              }
            }
            if ( *(_DWORD *)(v7 + 32) == v8 && *(_DWORD *)(v7 + 36) < *(_DWORD *)(v5 + 36) )
            {
              v7 = *(_QWORD *)(v7 + 24);
              goto LABEL_12;
            }
            v9 = v7;
            v7 = *(_QWORD *)(v7 + 16);
            if ( !v7 )
              goto LABEL_13;
          }
        }
        while ( 1 )
        {
          v9 = (__int64)v3;
LABEL_20:
          v20[0] = (const __m128i *)(v6 + 1);
          v19 = (_QWORD *)(v5 + 32);
          sub_26E38E0(a3, v9, &v19, v20);
LABEL_17:
          v6 = (__int64 *)*v6;
          if ( !v6 )
            break;
          v7 = a3[2];
          if ( v7 )
            goto LABEL_7;
        }
      }
    }
    v5 = sub_220EF30(v5);
  }
  v10 = *(_QWORD *)(a2 + 144);
  result = a2 + 128;
  v17 = result;
  if ( result != v10 )
  {
    do
    {
      if ( *(char *)(v10 + 33) >= 0 )
      {
        v12 = *(const __m128i **)(v10 + 64);
        if ( (const __m128i *)(v10 + 48) != v12 )
        {
          v13 = a3[2];
          if ( v13 )
          {
LABEL_31:
            v14 = *(_DWORD *)(v10 + 32);
            v15 = (__int64)(a3 + 1);
            while ( 1 )
            {
              while ( *(_DWORD *)(v13 + 32) < v14 )
              {
                v13 = *(_QWORD *)(v13 + 24);
LABEL_36:
                if ( !v13 )
                {
LABEL_37:
                  if ( a3 + 1 != (_QWORD *)v15
                    && *(_DWORD *)(v15 + 32) <= v14
                    && (*(_DWORD *)(v15 + 32) != v14 || *(_DWORD *)(v10 + 36) >= *(_DWORD *)(v15 + 36)) )
                  {
                    *(_QWORD *)(v15 + 48) = 23;
                    *(_QWORD *)(v15 + 40) = "unknown.indirect.callee";
                    goto LABEL_41;
                  }
                  goto LABEL_44;
                }
              }
              if ( *(_DWORD *)(v13 + 32) == v14 && *(_DWORD *)(v13 + 36) < *(_DWORD *)(v10 + 36) )
              {
                v13 = *(_QWORD *)(v13 + 24);
                goto LABEL_36;
              }
              v15 = v13;
              v13 = *(_QWORD *)(v13 + 16);
              if ( !v13 )
                goto LABEL_37;
            }
          }
          while ( 1 )
          {
            v15 = (__int64)(a3 + 1);
LABEL_44:
            v20[0] = v12 + 2;
            v19 = (_QWORD *)(v10 + 32);
            sub_26E38E0(a3, v15, &v19, v20);
LABEL_41:
            v12 = (const __m128i *)sub_220EF30((__int64)v12);
            if ( (const __m128i *)(v10 + 48) == v12 )
              break;
            v13 = a3[2];
            if ( v13 )
              goto LABEL_31;
          }
        }
      }
      result = sub_220EF30(v10);
      v10 = result;
    }
    while ( v17 != result );
  }
  return result;
}
