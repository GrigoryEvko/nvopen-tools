// Function: sub_1454560
// Address: 0x1454560
//
char __fastcall sub_1454560(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // r13
  _QWORD *v5; // r14
  __int64 v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // r13
  char result; // al
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // rsi
  __int64 v15; // r8
  _QWORD *v16; // rbx
  __int64 v17; // r12
  _QWORD *v18; // r14
  __int64 v19; // rax
  __int64 v20; // r12
  _QWORD *v21; // r12
  int v22; // esi
  int v23; // r10d

  if ( *(_DWORD *)(a2 + 32) )
  {
    v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 32LL))(a2);
    v11 = *(unsigned int *)(a1 + 208);
    if ( (_DWORD)v11 )
    {
      v12 = *(_QWORD *)(a1 + 192);
      v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v14 = (__int64 *)(v12 + 56LL * v13);
      v15 = *v14;
      if ( v10 != *v14 )
      {
        v22 = 1;
        while ( v15 != -8 )
        {
          v23 = v22 + 1;
          v13 = (v11 - 1) & (v22 + v13);
          v14 = (__int64 *)(v12 + 56LL * v13);
          v15 = *v14;
          if ( v10 == *v14 )
            goto LABEL_13;
          v22 = v23;
        }
        return 0;
      }
LABEL_13:
      if ( v14 != (__int64 *)(v12 + 56 * v11) )
      {
        v16 = (_QWORD *)v14[1];
        v17 = 8LL * *((unsigned int *)v14 + 4);
        v18 = &v16[(unsigned __int64)v17 / 8];
        v19 = v17 >> 3;
        v20 = v17 >> 5;
        if ( v20 )
        {
          v21 = &v16[4 * v20];
          while ( 1 )
          {
            if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v16 + 16LL))(*v16, a2) )
              return v16 != v18;
            if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v16[1] + 16LL))(v16[1], a2) )
              return v18 != v16 + 1;
            if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v16[2] + 16LL))(v16[2], a2) )
              return v18 != v16 + 2;
            if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v16[3] + 16LL))(v16[3], a2) )
              return v18 != v16 + 3;
            v16 += 4;
            if ( v21 == v16 )
            {
              v19 = v18 - v16;
              break;
            }
          }
        }
        if ( v19 != 2 )
        {
          if ( v19 != 3 )
          {
            if ( v19 == 1 )
            {
LABEL_41:
              result = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v16 + 16LL))(*v16, a2);
              if ( !result )
                return result;
              return v18 != v16;
            }
            return 0;
          }
          if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v16 + 16LL))(*v16, a2) )
            return v18 != v16;
          ++v16;
        }
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)*v16 + 16LL))(*v16, a2) )
          return v18 != v16;
        ++v16;
        goto LABEL_41;
      }
    }
    return 0;
  }
  v3 = *(_QWORD **)(a2 + 40);
  v4 = 8LL * *(unsigned int *)(a2 + 48);
  v5 = &v3[(unsigned __int64)v4 / 8];
  v6 = v4 >> 3;
  v7 = v4 >> 5;
  if ( v7 )
  {
    v8 = &v3[4 * v7];
    while ( (unsigned __int8)sub_1454560(a1, *v3) )
    {
      if ( !(unsigned __int8)sub_1454560(a1, v3[1]) )
        return v5 == ++v3;
      if ( !(unsigned __int8)sub_1454560(a1, v3[2]) )
        return v5 == v3 + 2;
      if ( !(unsigned __int8)sub_1454560(a1, v3[3]) )
        return v5 == v3 + 3;
      v3 += 4;
      if ( v3 == v8 )
      {
        v6 = v5 - v3;
        goto LABEL_24;
      }
    }
    return v5 == v3;
  }
LABEL_24:
  if ( v6 == 2 )
    goto LABEL_33;
  if ( v6 == 3 )
  {
    if ( !(unsigned __int8)sub_1454560(a1, *v3) )
      return v5 == v3;
    ++v3;
LABEL_33:
    if ( (unsigned __int8)sub_1454560(a1, *v3) )
    {
      ++v3;
      goto LABEL_35;
    }
    return v5 == v3;
  }
  if ( v6 != 1 )
    return 1;
LABEL_35:
  result = sub_1454560(a1, *v3);
  if ( !result )
    return v5 == v3;
  return result;
}
