// Function: sub_A6FBB0
// Address: 0xa6fbb0
//
char *__fastcall sub_A6FBB0(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "none";
      break;
    case 1:
      result = "allocalign";
      break;
    case 2:
      result = "allocptr";
      break;
    case 3:
      result = "alwaysinline";
      break;
    case 4:
      result = "builtin";
      break;
    case 5:
      result = "cold";
      break;
    case 6:
      result = "convergent";
      break;
    case 7:
      result = "coro_only_destroy_when_complete";
      break;
    case 8:
      result = "coro_elide_safe";
      break;
    case 9:
      result = "dead_on_unwind";
      break;
    case 10:
      result = "disable_sanitizer_instrumentation";
      break;
    case 11:
      result = "fn_ret_thunk_extern";
      break;
    case 12:
      result = "hot";
      break;
    case 13:
      result = "hybrid_patchable";
      break;
    case 14:
      result = "immarg";
      break;
    case 15:
      result = "inreg";
      break;
    case 16:
      result = "inlinehint";
      break;
    case 17:
      result = "jumptable";
      break;
    case 18:
      result = "minsize";
      break;
    case 19:
      result = "mustprogress";
      break;
    case 20:
      result = "naked";
      break;
    case 21:
      result = "nest";
      break;
    case 22:
      result = "noalias";
      break;
    case 23:
      result = "nobuiltin";
      break;
    case 24:
      result = "nocallback";
      break;
    case 25:
      result = "nocf_check";
      break;
    case 26:
      result = "nodivergencesource";
      break;
    case 27:
      result = "noduplicate";
      break;
    case 28:
      result = "noext";
      break;
    case 29:
      result = "nofree";
      break;
    case 30:
      result = "noimplicitfloat";
      break;
    case 31:
      result = "noinline";
      break;
    case 32:
      result = "nomerge";
      break;
    case 33:
      result = "noprofile";
      break;
    case 34:
      result = "norecurse";
      break;
    case 35:
      result = "noredzone";
      break;
    case 36:
      result = "noreturn";
      break;
    case 37:
      result = "nosanitize_bounds";
      break;
    case 38:
      result = "nosanitize_coverage";
      break;
    case 39:
      result = "nosync";
      break;
    case 40:
      result = "noundef";
      break;
    case 41:
      result = "nounwind";
      break;
    case 42:
      result = "nonlazybind";
      break;
    case 43:
      result = "nonnull";
      break;
    case 44:
      result = "null_pointer_is_valid";
      break;
    case 45:
      result = "optforfuzzing";
      break;
    case 46:
      result = "optdebug";
      break;
    case 47:
      result = "optsize";
      break;
    case 48:
      result = "optnone";
      break;
    case 49:
      result = "presplitcoroutine";
      break;
    case 50:
      result = "readnone";
      break;
    case 51:
      result = "readonly";
      break;
    case 52:
      result = "returned";
      break;
    case 53:
      result = "returns_twice";
      break;
    case 54:
      result = "signext";
      break;
    case 55:
      result = "safestack";
      break;
    case 56:
      result = "sanitize_address";
      break;
    case 57:
      result = "sanitize_hwaddress";
      break;
    case 58:
      result = "sanitize_memtag";
      break;
    case 59:
      result = "sanitize_memory";
      break;
    case 60:
      result = "sanitize_numerical_stability";
      break;
    case 61:
      result = "sanitize_realtime";
      break;
    case 62:
      result = "sanitize_realtime_blocking";
      break;
    case 63:
      result = "sanitize_thread";
      break;
    case 64:
      result = "sanitize_type";
      break;
    case 65:
      result = "shadowcallstack";
      break;
    case 66:
      result = "skipprofile";
      break;
    case 67:
      result = "speculatable";
      break;
    case 68:
      result = "speculative_load_hardening";
      break;
    case 69:
      result = "ssp";
      break;
    case 70:
      result = "sspreq";
      break;
    case 71:
      result = "sspstrong";
      break;
    case 72:
      result = "strictfp";
      break;
    case 73:
      result = "swiftasync";
      break;
    case 74:
      result = "swifterror";
      break;
    case 75:
      result = "swiftself";
      break;
    case 76:
      result = "willreturn";
      break;
    case 77:
      result = "writable";
      break;
    case 78:
      result = "writeonly";
      break;
    case 79:
      result = "zeroext";
      break;
    case 80:
      result = "byref";
      break;
    case 81:
      result = "byval";
      break;
    case 82:
      result = "elementtype";
      break;
    case 83:
      result = "inalloca";
      break;
    case 84:
      result = "preallocated";
      break;
    case 85:
      result = "sret";
      break;
    case 86:
      result = "align";
      break;
    case 87:
      result = "allockind";
      break;
    case 88:
      result = "allocsize";
      break;
    case 89:
      result = "captures";
      break;
    case 90:
      result = "dereferenceable";
      break;
    case 91:
      result = "dereferenceable_or_null";
      break;
    case 92:
      result = "memory";
      break;
    case 93:
      result = "nofpclass";
      break;
    case 94:
      result = "alignstack";
      break;
    case 95:
      result = "uwtable";
      break;
    case 96:
      result = "vscale_range";
      break;
    case 97:
      result = "range";
      break;
    case 98:
      result = "initializes";
      break;
    default:
      BUG();
  }
  return result;
}
