// Function: sub_2ECAAF0
// Address: 0x2ecaaf0
//
__int64 __fastcall sub_2ECAAF0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  _BYTE *v5; // rdx
  _BYTE *v6; // r12
  _BYTE *v7; // rbx
  unsigned __int8 v8; // dl
  _BYTE *v9; // r15
  unsigned int v10; // r9d
  unsigned int v11; // eax
  __int64 v12; // rdi
  __int64 v13; // rdx
  _DWORD *v14; // rsi
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+18h] [rbp-68h]
  int v20; // [rsp+2Ch] [rbp-54h]
  __m128i v21; // [rsp+30h] [rbp-50h] BYREF
  __int64 v22; // [rsp+40h] [rbp-40h]
  __int64 *v23; // [rsp+48h] [rbp-38h]

  v3 = *(_QWORD *)(*a2 + 32);
  v17 = *a2;
  result = v3 + 40LL * (*(_DWORD *)(*a2 + 40) & 0xFFFFFF);
  v19 = result;
  if ( result != v3 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v3 )
        goto LABEL_8;
      result = *(unsigned __int8 *)(v3 + 4);
      if ( (result & 1) != 0 || (result & 2) != 0 )
        goto LABEL_8;
      if ( (*(_BYTE *)(v3 + 3) & 0x10) != 0 )
      {
        if ( (*(_DWORD *)v3 & 0xFFF00) == 0 )
          goto LABEL_8;
        result = a1;
        if ( *(_BYTE *)(a1 + 899) )
          goto LABEL_8;
        result = *(unsigned int *)(v3 + 8);
        v20 = *(_DWORD *)(v3 + 8);
LABEL_36:
        if ( v20 < 0 )
          goto LABEL_23;
        v3 += 40;
        if ( v19 == v3 )
          return result;
      }
      else
      {
        v20 = *(_DWORD *)(v3 + 8);
        result = a1;
        if ( !*(_BYTE *)(a1 + 899) )
          goto LABEL_36;
        result = (unsigned int)v20;
        if ( v20 < 0 )
        {
          v5 = *(_BYTE **)(v17 + 32);
          v6 = &v5[40 * (*(_DWORD *)(v17 + 40) & 0xFFFFFF)];
          if ( v5 != v6 )
          {
            while ( 1 )
            {
              v7 = v5;
              if ( sub_2DADC00(v5) )
                break;
              v5 = v7 + 40;
              if ( v6 == v7 + 40 )
                goto LABEL_23;
            }
            while ( v6 != v7 )
            {
              if ( v20 == *((_DWORD *)v7 + 2) )
              {
                v8 = v7[3];
                result = (v8 & 0x10) != 0;
                if ( ((unsigned __int8)result & (v8 >> 6)) == 0 )
                  goto LABEL_8;
              }
              if ( v7 + 40 == v6 )
                break;
              v9 = v7 + 40;
              while ( 1 )
              {
                v7 = v9;
                if ( sub_2DADC00(v9) )
                  break;
                v9 += 40;
                if ( v6 == v9 )
                  goto LABEL_23;
              }
            }
          }
LABEL_23:
          v10 = *(_DWORD *)(a1 + 3648);
          v11 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 3976) + (v20 & 0x7FFFFFFF));
          if ( v11 >= v10 )
            goto LABEL_34;
          v12 = *(_QWORD *)(a1 + 3640);
          while ( 1 )
          {
            v13 = v11;
            v14 = (_DWORD *)(v12 + 40LL * v11);
            if ( (v20 & 0x7FFFFFFF) == (*v14 & 0x7FFFFFFF) )
            {
              v15 = (unsigned int)v14[8];
              if ( (_DWORD)v15 != -1 && *(_DWORD *)(v12 + 40 * v15 + 36) == -1 )
                break;
            }
            v11 += 256;
            if ( v10 <= v11 )
              goto LABEL_34;
          }
          if ( v11 == -1 )
          {
LABEL_34:
            v23 = a2;
            v21.m128i_i64[1] = 0;
            v21.m128i_i32[0] = v20;
            v22 = 0;
            result = sub_2ECA8A0(a1 + 3640, &v21);
          }
          else
          {
            while ( 1 )
            {
              result = v12 + 40 * v13;
              if ( *(__int64 **)(result + 24) == a2 )
                break;
              v16 = *(_DWORD *)(result + 36);
              if ( v16 == -1 )
                goto LABEL_34;
              v13 = v16;
            }
          }
        }
LABEL_8:
        v3 += 40;
        if ( v19 == v3 )
          return result;
      }
    }
  }
  return result;
}
