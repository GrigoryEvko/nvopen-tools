// Function: sub_227AD80
// Address: 0x227ad80
//
__int64 __fastcall sub_227AD80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 *v9; // rax
  __int64 i; // rdx
  __int64 *v11; // r15
  __int64 v12; // r12
  __int64 *v13; // r14
  __int64 *v14; // r12
  __int64 result; // rax
  __int64 *j; // r13
  __int64 *v17; // rdi
  __int64 *v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rsi
  __int64 *v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // r13
  __int64 v25; // rcx
  __int64 v26; // rdi

  v7 = *(unsigned int *)(a2 + 68);
  if ( *(_DWORD *)(a2 + 72) != (_DWORD)v7 || (result = sub_B19060(a2, (__int64)&unk_4F82400, a3, v7), !(_BYTE)result) )
  {
    if ( *(_DWORD *)(a1 + 68) == *(_DWORD *)(a1 + 72)
      && (result = sub_B19060(a1, (__int64)&unk_4F82400, a3, v7), (_BYTE)result) )
    {
      if ( a1 != a2 )
        result = sub_C8CE00(a1, a1 + 32, a2, v25, a5, a6);
      v26 = a1 + 48;
      if ( a2 + 48 != a1 + 48 )
        return sub_C8CE00(v26, a1 + 80, a2 + 48, v25, a5, a6);
    }
    else
    {
      v8 = a2;
      v9 = *(__int64 **)(a2 + 56);
      if ( *(_BYTE *)(a2 + 76) )
      {
        i = *(unsigned int *)(a2 + 68);
      }
      else
      {
        v8 = a2;
        i = *(unsigned int *)(a2 + 64);
      }
      v11 = &v9[i];
      if ( v9 != v11 )
      {
        while ( 1 )
        {
          v12 = *v9;
          v13 = v9;
          if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v11 == ++v9 )
            goto LABEL_8;
        }
        if ( v9 != v11 )
        {
          if ( !*(_BYTE *)(a1 + 28) )
            goto LABEL_34;
LABEL_18:
          v17 = *(__int64 **)(a1 + 8);
          i = (__int64)&v17[*(unsigned int *)(a1 + 20)];
          v18 = v17;
          if ( v17 != (__int64 *)i )
          {
            while ( *v18 != v12 )
            {
              if ( (__int64 *)i == ++v18 )
                goto LABEL_23;
            }
            v19 = (unsigned int)(*(_DWORD *)(a1 + 20) - 1);
            *(_DWORD *)(a1 + 20) = v19;
            i = v17[v19];
            *v18 = i;
            ++*(_QWORD *)a1;
          }
LABEL_23:
          if ( *(_BYTE *)(a1 + 76) )
          {
LABEL_24:
            v20 = *(_QWORD **)(a1 + 56);
            v21 = *(unsigned int *)(a1 + 68);
            for ( i = (__int64)&v20[v21]; (_QWORD *)i != v20; ++v20 )
            {
              if ( *v20 == v12 )
                goto LABEL_28;
            }
            if ( (unsigned int)v21 < *(_DWORD *)(a1 + 64) )
            {
              *(_DWORD *)(a1 + 68) = v21 + 1;
              *(_QWORD *)i = v12;
              ++*(_QWORD *)(a1 + 48);
              goto LABEL_28;
            }
          }
          while ( 1 )
          {
            sub_C8CC70(a1 + 48, v12, i, v8, a5, a6);
LABEL_28:
            v22 = v13 + 1;
            if ( v13 + 1 == v11 )
              break;
            v12 = *v22;
            for ( ++v13; (unsigned __int64)*v22 >= 0xFFFFFFFFFFFFFFFELL; v13 = v22 )
            {
              if ( v11 == ++v22 )
                goto LABEL_8;
              v12 = *v22;
            }
            if ( v13 == v11 )
              break;
            if ( *(_BYTE *)(a1 + 28) )
              goto LABEL_18;
LABEL_34:
            v23 = sub_C8CA60(a1, v12);
            if ( !v23 )
              goto LABEL_23;
            *v23 = -2;
            ++*(_DWORD *)(a1 + 24);
            ++*(_QWORD *)a1;
            if ( *(_BYTE *)(a1 + 76) )
              goto LABEL_24;
          }
        }
      }
LABEL_8:
      v14 = *(__int64 **)(a1 + 8);
      if ( *(_BYTE *)(a1 + 28) )
      {
        result = *(unsigned int *)(a1 + 20);
        v24 = &v14[result];
        while ( v24 != v14 )
        {
          result = sub_B19060(a2, *v14, i, v8);
          if ( (_BYTE)result )
          {
            ++v14;
          }
          else
          {
            result = *--v24;
            *v14 = result;
            --*(_DWORD *)(a1 + 20);
            ++*(_QWORD *)a1;
          }
        }
      }
      else
      {
        result = *(unsigned int *)(a1 + 16);
        for ( j = &v14[result]; v14 != j; ++v14 )
        {
          if ( (unsigned __int64)*v14 < 0xFFFFFFFFFFFFFFFELL )
          {
            result = sub_B19060(a2, *v14, i, v8);
            if ( !(_BYTE)result )
            {
              *v14 = -2;
              ++*(_DWORD *)(a1 + 24);
              ++*(_QWORD *)a1;
            }
          }
        }
      }
    }
  }
  return result;
}
